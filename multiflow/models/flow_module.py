from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
import shutil
import torch.distributed as dist
from glob import glob
from pytorch_lightning import LightningModule
from multiflow.analysis import utils as au
from multiflow.models.flow_model import FlowModel
from multiflow.models import utils as mu
from multiflow.models import folding_model
from multiflow.data.interpolant import Interpolant 
from multiflow.data import utils as du
from multiflow.data import all_atom, so3_utils
from multiflow.data.residue_constants import restypes, restypes_with_x
from multiflow.data import residue_constants
from multiflow.experiments import utils as eu
from biotite.sequence.io import fasta
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


class FlowModule(LightningModule):

    def __init__(self, cfg, dataset_cfg, folding_cfg=None, folding_device_id=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._dataset_cfg = dataset_cfg
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

        self._folding_model = None
        self._folding_cfg = folding_cfg
        self._folding_device_id = folding_device_id

        self.aatype_pred_num_tokens = cfg.model.aatype_pred_num_tokens

    @property
    def folding_model(self):
        if self._folding_model is None:
            self._folding_model = folding_model.FoldingModel(
                self._folding_cfg,
                device_id=self._folding_device_id
            )
        return self._folding_model

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()


    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if training_cfg.mask_plddt:
            loss_mask *= noisy_batch['plddt_mask']
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_aatypes_1 = noisy_batch['aatypes_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 

        # Timestep used for normalization.
        r3_t = noisy_batch['r3_t'] # (B, 1)
        so3_t = noisy_batch['so3_t'] # (B, 1)
        cat_t = noisy_batch['cat_t'] # (B, 1)
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
        if training_cfg.aatypes_loss_use_likelihood_weighting:
            cat_norm_scale = 1 - torch.min(
                cat_t, torch.tensor(training_cfg.t_normalize_clip)) # (B, 1)
            assert cat_norm_scale.shape == (num_batch, 1)
        else:
            cat_norm_scale = 1.0
        
        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_logits = model_output['pred_logits'] # (B, N, aatype_pred_num_tokens)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError('NaN encountered in pred_rots_vf')
        
        # aatypes loss
        ce_loss = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, self.aatype_pred_num_tokens),
            gt_aatypes_1.flatten().long(),
            reduction='none',
        ).reshape(num_batch, num_res) / cat_norm_scale
        aatypes_loss = torch.sum(ce_loss * loss_mask, dim=-1) / (loss_denom / 3)
        aatypes_loss *= training_cfg.aatypes_loss_weight

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        trans_loss = torch.clamp(trans_loss, max=5)

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (
            bb_atom_loss * training_cfg.aux_loss_use_bb_loss
            + dist_mat_loss * training_cfg.aux_loss_use_pair_loss
        )
        auxiliary_loss *= (
            (r3_t[:, 0] > training_cfg.aux_loss_t_pass)
            & (so3_t[:, 0] > training_cfg.aux_loss_t_pass)
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)

        train_loss = trans_loss + rots_vf_loss + auxiliary_loss + aatypes_loss
        if torch.any(torch.isnan(train_loss)):
            raise ValueError('NaN loss encountered')
        self._prev_batch = noisy_batch
        self._prev_loss_denom = loss_denom
        self._prev_loss = {
            "trans_loss": trans_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "train_loss": train_loss,
            'aatypes_loss': aatypes_loss
        }
        return self._prev_loss

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        
        diffuse_mask = batch['diffuse_mask']
        csv_idx = batch['csv_idx']

        assert (diffuse_mask == 1.0).all()

        prot_traj, model_traj = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            trans_1=batch['trans_1'],
            rotmats_1=batch['rotmats_1'],
            aatypes_1=batch['aatypes_1'],
            diffuse_mask=diffuse_mask,
            chain_idx=batch['chain_idx'],
            res_idx=batch['res_idx'],
        )
        samples = prot_traj[-1][0].numpy()
        assert samples.shape == (num_batch, num_res, 37, 3)
        # assert False, "need to separate aatypes from atom37_traj"

        generated_aatypes = prot_traj[-1][1].numpy()
        assert generated_aatypes.shape == (num_batch, num_res)
        batch_level_aatype_metrics = mu.calc_aatype_metrics(generated_aatypes)

        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f'sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}'
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(sample_dir, 'sample.pdb'),
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            # Run designability
            pmpnn_pdb_path = saved_path.replace('.pdb', '_pmpnn.pdb')
            shutil.copy(saved_path, pmpnn_pdb_path)
            pmpnn_fasta_path = self.run_pmpnn(
                sample_dir,
                pmpnn_pdb_path,
            )
            folded_dir = os.path.join(sample_dir, 'folded')
            os.makedirs(folded_dir, exist_ok=True)

            if self.interpolant._aatypes_cfg.corrupt:
                # Codesign
                codesign_fasta = fasta.FastaFile()
                codesign_fasta['codesign_seq_1'] = "".join([restypes_with_x[x] for x in generated_aatypes[i]])
                codesign_fasta_path = os.path.join(sample_dir, 'codesign.fa')
                codesign_fasta.write(codesign_fasta_path)

                codesign_folded_output = self.folding_model.fold_fasta(codesign_fasta_path, folded_dir)
                codesign_results = mu.process_folded_outputs(saved_path, codesign_folded_output)

                # make a fasta file with a single PMPNN sequence to be folded
                reloaded_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
                single_fasta = fasta.FastaFile()
                single_fasta['pmpnn_seq_1'] = reloaded_fasta['pmpnn_seq_1']
                single_fasta_path = os.path.join(sample_dir, 'pmpnn_single.fasta')
                single_fasta.write(single_fasta_path)

                single_pmpnn_folded_output = self.folding_model.fold_fasta(single_fasta_path, folded_dir)
                single_pmpnn_results = mu.process_folded_outputs(saved_path, single_pmpnn_folded_output)

                designable_metrics = {
                    'codesign_bb_rmsd': codesign_results.bb_rmsd.min(),
                    'pmpnn_bb_rmsd': single_pmpnn_results.bb_rmsd.min(),
                }
            else:
                # Just structure
                folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)

                designable_results = mu.process_folded_outputs(saved_path, folded_output) 
                designable_metrics = {
                    'bb_rmsd': designable_results.bb_rmsd.min()
                }
            try:
                mdtraj_metrics = mu.calc_mdtraj_metrics(saved_path)
                ca_ca_metrics = mu.calc_ca_ca_metrics(final_pos[:, residue_constants.atom_order['CA']])
                batch_metrics.append((mdtraj_metrics | ca_ca_metrics | designable_metrics | batch_level_aatype_metrics))
            except Exception as e:
                print(e)
                continue

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = (
                    model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None]
                    + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
                )
                logits_1 = torch.nn.functional.one_hot(
                    batch['aatypes_1'].long(), num_classes=self.aatype_pred_num_tokens).float()
                noisy_batch['aatypes_sc'] = (
                    model_sc['pred_logits'] * noisy_batch['diffuse_mask'][..., None]
                    + logits_1 * (1 - noisy_batch['diffuse_mask'][..., None])
                )
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        so3_t = torch.squeeze(noisy_batch['so3_t'])
        self._log_scalar(
            "train/so3_t",
            np.mean(du.to_numpy(so3_t)),
            prog_bar=False, batch_size=num_batch)
        r3_t = torch.squeeze(noisy_batch['r3_t'])
        self._log_scalar(
            "train/r3_t",
            np.mean(du.to_numpy(r3_t)),
            prog_bar=False, batch_size=num_batch)
        cat_t = torch.squeeze(noisy_batch['cat_t'])
        self._log_scalar(
            "train/cat_t",
            np.mean(du.to_numpy(cat_t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == 'rots_vf_loss':
                batch_t = so3_t
            elif loss_name == 'train_loss':
                continue
            elif loss_name == 'aatypes_loss':
                batch_t = cat_t
            else:
                batch_t = r3_t
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = total_losses['train_loss']
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )

    def predict_step(self, batch, batch_idx):
        del batch_idx # Unused
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        if 'sample_id' in batch:
            sample_ids = batch['sample_id'].squeeze().tolist()
        else:
            sample_ids = [0]
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)

        if self._infer_cfg.task == 'unconditional':
            sample_length = batch['num_res'].item()
            true_bb_pos = None
            sample_dirs = [os.path.join(
                self.inference_dir, f'length_{sample_length}', f'sample_{str(sample_id)}')
                for sample_id in sample_ids]
            trans_1 = rotmats_1 = diffuse_mask = aatypes_1 = true_aatypes = None
        elif self._infer_cfg.task == 'forward_folding':
            sample_length = batch['trans_1'].shape[1]
            sample_dirs = [os.path.join(self.inference_dir, f'length_{sample_length}', batch['pdb_name'][0])]
            for sample_dir in sample_dirs:
                os.makedirs(sample_dir, exist_ok=True)
            true_bb_pos = all_atom.atom37_from_trans_rot(batch['trans_1'], batch['rotmats_1'])
            assert true_bb_pos.shape == (1, sample_length, 37, 3)
            # save the ground truth as a pdb
            au.write_prot_to_pdb(
                prot_pos=true_bb_pos[0].cpu().detach().numpy(),
                file_path=os.path.join(sample_dirs[0], batch['pdb_name'][0] + '_gt.pdb'),
                aatype=batch['aatypes_1'][0].cpu().detach().numpy(),
            )
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy() 
            assert true_bb_pos.shape == (sample_length * 3, 3)
            aatypes_1 = batch['aatypes_1']
            trans_1 = rotmats_1 = diffuse_mask = true_aatypes = None
        elif self._infer_cfg.task == 'inverse_folding':
            sample_length = batch['trans_1'].shape[1]
            trans_1 = batch['trans_1']
            rotmats_1 = batch['rotmats_1']
            true_aatypes = batch['aatypes_1']
            sample_dirs = [os.path.join(self.inference_dir, f'length_{sample_length}', batch['pdb_name'][0])]
            aatypes_1 = diffuse_mask = true_bb_pos = None
        else:
            raise ValueError(f'Unknown task {self._infer_cfg.task}')

        # Skip runs if already exist
        top_sample_csv_paths = [os.path.join(sample_dir, 'top_sample.csv')
                                for sample_dir in sample_dirs]
        if all([os.path.exists(top_sample_csv_path) for top_sample_csv_path in top_sample_csv_paths]):
            self._print_logger.info(f'Skipping instance {sample_ids} length {sample_length}')
            return
        # Sample batch
        prot_traj, model_traj = interpolant.sample(
            num_batch, sample_length, self.model,
            trans_1=trans_1, rotmats_1=rotmats_1, aatypes_1=aatypes_1,
            diffuse_mask=diffuse_mask,
            forward_folding=self._infer_cfg.task == 'forward_folding',
            inverse_folding=self._infer_cfg.task == 'inverse_folding',
            separate_t=self._infer_cfg.interpolant.codesign_separate_t,
        )
        diffuse_mask = diffuse_mask if diffuse_mask is not None else torch.ones(1, sample_length)
        atom37_traj = [x[0] for x in prot_traj]
        atom37_model_traj = [x[0] for x in model_traj]

        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        noisy_traj_length = bb_trajs.shape[1]
        assert bb_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)

        model_trajs = du.to_numpy(torch.stack(atom37_model_traj, dim=0).transpose(0, 1))
        clean_traj_length = model_trajs.shape[1]
        assert model_trajs.shape == (num_batch, clean_traj_length, sample_length, 37, 3)

        aa_traj = [x[1] for x in prot_traj]
        clean_aa_traj = [x[1] for x in model_traj]

        aa_trajs = du.to_numpy(torch.stack(aa_traj, dim=0).transpose(0, 1).long())
        assert aa_trajs.shape == (num_batch, noisy_traj_length, sample_length)

        for i in range(aa_trajs.shape[0]):
            for j in range(aa_trajs.shape[2]):
                if aa_trajs[i, -1, j] == du.MASK_TOKEN_INDEX:
                    print("WARNING mask in predicted AA")
                    aa_trajs[i, -1, j] = 0
        clean_aa_trajs = du.to_numpy(torch.stack(clean_aa_traj, dim=0).transpose(0, 1).long())
        assert clean_aa_trajs.shape == (num_batch, clean_traj_length, sample_length)

        for i, sample_id in zip(range(num_batch), sample_ids):
            sample_dir = sample_dirs[i]
            top_sample_df = self.compute_sample_metrics(
                batch,
                model_trajs[i],
                bb_trajs[i],
                aa_trajs[i],
                clean_aa_trajs[i],
                true_bb_pos,
                true_aatypes,
                diffuse_mask,
                sample_id,
                sample_length,
                sample_dir,
                interpolant._aatypes_cfg.corrupt,
                self._infer_cfg.also_fold_pmpnn_seq,
                self._infer_cfg.write_sample_trajectories,
            )
            top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
            top_sample_df.to_csv(top_sample_csv_path)


    def run_pmpnn(
            self,
            write_dir,
            pdb_input_path,
        ):
        self.folding_model.run_pmpnn(
            write_dir,
            pdb_input_path,
        )
        mpnn_fasta_path = os.path.join(
            write_dir,
            'seqs',
            os.path.basename(pdb_input_path).replace('.pdb', '.fa')
        )
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        all_header_seqs = [
            (f'pmpnn_seq_{i}', seq) for i, (_, seq) in enumerate(fasta_seqs.items())
            if i > 0
        ]
        modified_fasta_path = mpnn_fasta_path.replace('.fa', '_modified.fasta')
        fasta.FastaFile.write_iter(modified_fasta_path, all_header_seqs)
        return modified_fasta_path


    def compute_sample_metrics(self, batch, model_traj, bb_traj, aa_traj,
                               clean_aa_traj, true_bb_pos, true_aa, diffuse_mask,
                               sample_id, sample_length, sample_dir,
                               aatypes_corrupt,
                               also_fold_pmpnn_seq, write_sample_trajectories):

        noisy_traj_length, sample_length, _, _ = bb_traj.shape
        clean_traj_length = model_traj.shape[0]
        assert bb_traj.shape == (noisy_traj_length, sample_length, 37, 3)
        assert model_traj.shape == (clean_traj_length, sample_length, 37, 3)
        assert aa_traj.shape == (noisy_traj_length, sample_length)
        assert clean_aa_traj.shape == (clean_traj_length, sample_length)


        os.makedirs(sample_dir, exist_ok=True)

        traj_paths = eu.save_traj(
            bb_traj[-1],
            bb_traj,
            np.flip(model_traj, axis=0),
            du.to_numpy(diffuse_mask)[0],
            output_dir=sample_dir,
            aa_traj=aa_traj, 
            clean_aa_traj = clean_aa_traj,
            write_trajectories=write_sample_trajectories,
        )

        # Run PMPNN to get sequences
        sc_output_dir = os.path.join(sample_dir, 'self_consistency')
        os.makedirs(sc_output_dir, exist_ok=True)
        pdb_path = traj_paths['sample_path']
        pmpnn_pdb_path = os.path.join(
            sc_output_dir, os.path.basename(pdb_path))
        shutil.copy(pdb_path, pmpnn_pdb_path)
        assert (diffuse_mask == 1.0).all()
        pmpnn_fasta_path = self.run_pmpnn(
            sc_output_dir,
            pmpnn_pdb_path,
        )

        os.makedirs(os.path.join(sc_output_dir, 'codesign_seqs'), exist_ok=True)
        codesign_fasta = fasta.FastaFile()
        codesign_fasta['codesign_seq_1'] = "".join([restypes[x] for x in aa_traj[-1]])
        codesign_fasta_path = os.path.join(sc_output_dir, 'codesign_seqs', 'codesign.fa')
        codesign_fasta.write(codesign_fasta_path)


        folded_dir = os.path.join(sc_output_dir, 'folded')
        if os.path.exists(folded_dir):
            shutil.rmtree(folded_dir)
        os.makedirs(folded_dir, exist_ok=False)
        if aatypes_corrupt:
            # codesign metrics
            folded_output = self.folding_model.fold_fasta(codesign_fasta_path, folded_dir)

            if also_fold_pmpnn_seq:
                pmpnn_folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)
                pmpnn_results = mu.process_folded_outputs(pdb_path, pmpnn_folded_output, true_bb_pos)
                pmpnn_results.to_csv(os.path.join(sample_dir, 'pmpnn_results.csv'))

        else:
            # non-codesign metrics
            folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)

        mpnn_results = mu.process_folded_outputs(pdb_path, folded_output, true_bb_pos)


        if true_aa is not None:
            assert true_aa.shape == (1, sample_length)

            true_aa_fasta = fasta.FastaFile()
            true_aa_fasta['seq_1'] = "".join([restypes_with_x[i] for i in true_aa[0]])
            true_aa_fasta.write(os.path.join(sample_dir, 'true_aa.fa'))

            seq_recovery = (torch.from_numpy(aa_traj[-1]).to(true_aa[0].device) == true_aa[0]).float().mean()
            mpnn_results['inv_fold_seq_recovery'] = seq_recovery.item()

            # get seq recovery for PMPNN as well
            pmpnn_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
            pmpnn_fasta_str = pmpnn_fasta['pmpnn_seq_1']
            pmpnn_fasta_idx = torch.tensor([restypes_with_x.index(x) for x in pmpnn_fasta_str]).to(true_aa[0].device)
            pmpnn_seq_recovery = (pmpnn_fasta_idx == true_aa[0]).float().mean()
            pmpnn_results['pmpnn_seq_recovery'] = pmpnn_seq_recovery.item()
            pmpnn_results.to_csv(os.path.join(sample_dir, 'pmpnn_results.csv'))
            mpnn_results['pmpnn_seq_recovery'] = pmpnn_seq_recovery.item()
            mpnn_results['pmpnn_bb_rmsd'] = pmpnn_results['bb_rmsd']

        # Save results to CSV
        mpnn_results.to_csv(os.path.join(sample_dir, 'sc_results.csv'))
        mpnn_results['length'] = sample_length
        mpnn_results['sample_id'] = sample_id
        del mpnn_results['header']
        del mpnn_results['sequence']

        # Select the top sample
        top_sample = mpnn_results.sort_values('bb_rmsd', ascending=True).iloc[:1]

        # Compute secondary structure metrics
        sample_dict = top_sample.iloc[0].to_dict()
        ss_metrics = mu.calc_mdtraj_metrics(sample_dict['sample_path'])
        top_sample['helix_percent'] = ss_metrics['helix_percent']
        top_sample['strand_percent'] = ss_metrics['strand_percent']
        return top_sample