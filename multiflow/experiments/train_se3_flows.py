import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from multiflow.data.datasets import PdbDataset
from multiflow.data.protein_dataloader import ProteinData
from multiflow.models.flow_module import FlowModule
from multiflow.experiments import utils as eu
import wandb
from pytorch_lightning.utilities import rank_zero_only

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')


class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._task = self._data_cfg.task
        self._dataset_cfg = self._setup_dataset()
        self._datamodule: LightningDataModule = ProteinData(
            data_cfg=self._data_cfg,
            dataset_cfg=self._dataset_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset
        )
        total_devices = self._exp_cfg.num_devices
        if self._cfg.folding.own_device:
            total_devices += 1
        device_ids = eu.get_available_device(total_devices)
        if self._cfg.folding.own_device:
            folding_device_id = device_ids[0]
            self._train_device_ids = device_ids[1:]
            log.info(f'Folding device id: {folding_device_id}')
        else:
            folding_device_id = None
            self._train_device_ids = device_ids
        log.info(f"Training with devices: {self._train_device_ids}")
        self._module: LightningModule = FlowModule(
            self._cfg,
            self._dataset_cfg,
            folding_cfg=self._cfg.folding,
            folding_device_id=folding_device_id,
        )
        if self._exp_cfg.raw_state_dict_reload is not None:
            self._module.load_state_dict(torch.load(self._exp_cfg.raw_state_dict_reload)['state_dict'])

        # Give model access to datamodule for post DDP setup processing.
        self._module._datamodule = self._datamodule

    def _setup_dataset(self):

        @rank_zero_only
        def create_synthetic_data_folder(folder_path): 
            os.makedirs(folder_path, exist_ok=False)

        if self._data_cfg.dataset == 'pdb':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                PdbDataset, self._cfg.pdb_dataset, self._task)
            dataset_cfg = self._cfg.pdb_dataset
        else:
            raise ValueError(f'Unrecognized dataset {self._data_cfg.dataset}') 

        return dataset_cfg
        
    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._train_device_ids = [self._train_device_ids[0]]
            self._data_cfg.loader.num_workers = 0
        else:
            logger = WandbLogger(
                **self._exp_cfg.wandb,
            )
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            
            # Save config only for main process.
            local_rank = os.environ.get('LOCAL_RANK', 0)
            if local_rank == 0:
                ckpt_dir = self._exp_cfg.checkpointer.dirpath
                log.info(f"Checkpoints saved to {ckpt_dir}")
                os.makedirs(ckpt_dir, exist_ok=True)
                cfg_path = os.path.join(ckpt_dir, 'config.yaml')
                with open(cfg_path, 'w') as f:
                    OmegaConf.save(config=self._cfg, f=f.name)
                cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
                flat_cfg = dict(eu.flatten_dict(cfg_dict))
                if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                    logger.experiment.config.update(flat_cfg)
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids,
        )
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )


@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()
