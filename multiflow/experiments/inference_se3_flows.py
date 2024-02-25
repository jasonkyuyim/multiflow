"""Script for running inference and evaluation.

To run inference on unconditional backbone generation:
> python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional

To run inference on inverse folding
> python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_inverse_folding

To run inference on forward folding
> python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_forward_folding

##########
# Config #
##########

Config locations:
- configs/inference_unconditional.yaml: unconditional sampling config.
- configs/inference_forward_folding.yaml: forward folding sampling config.
- configs/inference_inverse_folding.yaml: inverse folding sampling config.

Most important fields:
- inference.num_gpus: Number of GPUs to use. I typically use 2 or 4.

- inference.unconditional_ckpt_path: Checkpoint path for hallucination.
- inference.forward_folding_ckpt_path: Checkpoint path for forward folding.
- inference.inverse_folding_ckpt_path: Checkpoint path for inverse folding.

- inference.interpolant.sampling.num_timesteps: Number of steps in the flow.

- inference.folding.folding_model: `esmf` for ESMFold and `af2` for AlphaFold2.

[Only for hallucination]
- inference.samples.samples_per_length: Number of samples per length.
- inference.samples.min_length: Start of length range to sample.
- inference.samples.max_length: End of length range to sample.
- inference.samples.length_subset: Subset of lengths to sample. Will override min_length and max_length.

#######################
# Directory structure #
#######################

inference_outputs/                      # Inference run name. Same as Wandb run.
├── config.yaml                         # Inference and model config.
├── length_N                            # Directory for samples of length N.
│   ├── sample_X                        # Directory for sample X of length N.
│   │   ├── bb_traj.pdb                 # Flow matching trajectory
│   │   ├── sample.pdb                  # Final sample (final step of trajectory).
│   │   ├── self_consistency            # Directory of SC intermediate files.
│   │   │   ├── codesign_seqs           # Directory with codesign sequence
│   │   │   ├── folded                  # Directory with folded structures for ProteinMPNN and the Codesign Seq.
│   │   │   ├── esmf                    # Directory of ESMFold outputs.
│   │   │   ├── parsed_pdbs.jsonl       # ProteinMPNN compatible data file.
│   │   │   ├── sample.pdb              # Copy of sample_x/sample.pdb to use in ProteinMPNN
│   │   │   └── seqs                    # Directory of ProteinMPNN sequences.
│   │   │       └── sample.fa           # FASTA file of ProteinMPNN sequences.
│   │   ├── top_sample.csv              # CSV of the SC metrics for the best sequences and ESMFold structure.
│   │   ├── sc_results.csv              # All SC results from ProteinMPNN/ESMFold.
│   │   ├── pmpnn_results.csv           # Results from running ProteinMPNN on the structure.
│   │   └── x0_traj.pdb                 # Model x0 trajectory.


"""

import os
import time
import numpy as np
import hydra
import torch
import pandas as pd
import glob
import GPUtil
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from multiflow.experiments import utils as eu
from multiflow.models.flow_module import FlowModule
from multiflow.data.datasets import PdbDataset
import torch.distributed as dist


torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)


class EvalRunner:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """

        # Read in checkpoint.
        if cfg.inference.task == 'unconditional':
            ckpt_path = cfg.inference.unconditional_ckpt_path
        elif cfg.inference.task == 'forward_folding':
            ckpt_path = cfg.inference.forward_folding_ckpt_path
        elif cfg.inference.task == 'inverse_folding':
            ckpt_path = cfg.inference.inverse_folding_ckpt_path
        else:
            raise ValueError(f'Unknown task {cfg.inference.task}')
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))
        self._original_cfg = cfg.copy()

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up output directory only on rank 0
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            inference_dir = self.setup_inference_dir(ckpt_path)
            self._exp_cfg.inference_dir = inference_dir
            config_path = os.path.join(inference_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
            dataset_cfg=eu.get_dataset_cfg(cfg),
            folding_cfg=self._infer_cfg.folding,
        )
        log.info(pl.utilities.model_summary.ModelSummary(self._flow_module))
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg

    @property
    def inference_dir(self):
        return self._flow_module.inference_dir

    def setup_inference_dir(self, ckpt_path):
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        output_dir = os.path.join(
            self._infer_cfg.predict_dir,
            self._ckpt_name,
            self._infer_cfg.task,
            self._infer_cfg.inference_subdir,
        )
        os.makedirs(output_dir, exist_ok=True)
        log.info(f'Saving results to {output_dir}')
        return output_dir

    def run_sampling(self):
        devices = GPUtil.getAvailable(
            order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        log.info(f'Evaluating {self._infer_cfg.task}')
        if self._infer_cfg.task == 'unconditional':
            eval_dataset = eu.LengthDataset(self._samples_cfg)
        elif self._infer_cfg.task == 'forward_folding' or self._infer_cfg.task == 'inverse_folding':
            # We want to use the inference settings for the pdb dataset, not what was in the ckpt config
            self._cfg.pdb_post2021_dataset = self._original_cfg.pdb_post2021_dataset
            eval_dataset, _ = eu.dataset_creation(
                PdbDataset, self._cfg.pdb_post2021_dataset, 'hallucination'
            )
        else:
            raise ValueError(f'Unknown task {self._infer_cfg.task}')
        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
        )
        trainer.predict(self._flow_module, dataloaders=dataloader)

    def compute_unconditional_metrics(self, output_dir):
        log.info(f'Calculating metrics for {output_dir}')
        top_sample_csv = eu.get_all_top_samples(output_dir)
        top_sample_csv['designable'] = top_sample_csv.bb_rmsd <= 2.0
        metrics_df = pd.DataFrame(data={ 
            'Total codesignable': top_sample_csv.designable.sum(),
            'Designable': top_sample_csv.designable.mean(),
            'Total samples': len(top_sample_csv),
        }, index=[0])
        designable_csv_path = os.path.join(output_dir, 'designable.csv')
        metrics_df.to_csv(designable_csv_path, index=False)
        eu.calculate_diversity(
            output_dir, metrics_df, top_sample_csv, designable_csv_path)
        if self._infer_cfg.interpolant.aatypes.corrupt:
            # co-design metrics
            eu.calculate_pmpnn_consistency(output_dir, metrics_df, designable_csv_path)
            eu.calculate_pmpnn_designability(output_dir, metrics_df, designable_csv_path)

    def compute_forward_folding_metrics(self, output_dir):
        log.info(f'Calculating metrics for {output_dir}')
        top_sample_csv = eu.get_all_top_samples(output_dir)
        top_sample_csv['fold_match_seq'] = top_sample_csv.bb_rmsd_to_gt <= 2.0
        metrics_df = pd.DataFrame(data={ 
            'Total Match Seq': top_sample_csv.fold_match_seq.sum(),
            'Prop Match Seq': top_sample_csv.fold_match_seq.mean(),
            'Average bb_rmsd_to_gt': top_sample_csv.bb_rmsd_to_gt.mean(),
            'Average fold model bb_rmsd_to_gt': top_sample_csv.fold_model_bb_rmsd_to_gt.mean(),
            'Total samples': len(top_sample_csv),
        }, index=[0])
        metrics_csv_path = os.path.join(output_dir, 'forward_fold_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)

    def compute_inverse_folding_metrics(self, output_dir):
        log.info(f'Calculating metrics for {output_dir}')
        top_sample_csv = eu.get_all_top_samples(output_dir)
        top_sample_csv['designable'] = top_sample_csv.bb_rmsd <= 2.0
        metrics_df = pd.DataFrame(data={ 
            'Total designable': top_sample_csv.designable.sum(),
            'Designable': top_sample_csv.designable.mean(),
            'Total samples': len(top_sample_csv),
            'Average_bb_rmsd': top_sample_csv.bb_rmsd.mean(),
            'Average_seq_recovery': top_sample_csv.inv_fold_seq_recovery.mean(),
            'Average_pmpnn_bb_rmsd': top_sample_csv.pmpnn_bb_rmsd.mean(),
            'Average_pmpnn_seq_recovery': top_sample_csv.pmpnn_seq_recovery.mean(),
        }, index=[0])
        metrics_csv_path = os.path.join(output_dir, 'inverse_fold_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)



@hydra.main(version_base=None, config_path="../configs", config_name="inference_unconditional")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_sampling()
    
    def compute_metrics():
        if cfg.inference.task == 'unconditional':
            sampler.compute_unconditional_metrics(sampler.inference_dir)
        elif cfg.inference.task == 'forward_folding':
            sampler.compute_forward_folding_metrics(sampler.inference_dir)
        elif cfg.inference.task == 'inverse_folding':
            sampler.compute_inverse_folding_metrics(sampler.inference_dir)
        else:
            raise ValueError(f'Unknown task {cfg.inference.task}')

    if dist.is_initialized():
        if dist.get_rank() == 0:
            compute_metrics()
    else:
        compute_metrics()

    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()