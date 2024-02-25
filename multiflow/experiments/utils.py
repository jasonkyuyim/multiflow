"""Utility functions for experiments."""
import os
import numpy as np
import random
import torch
import glob
import re
import GPUtil
import shutil
import subprocess
import pandas as pd
import torch.distributed as dist
from openfold.utils import rigid_utils
import logging
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from multiflow.analysis import utils as au
from openfold.utils import rigid_utils as ru
from biotite.sequence.io import fasta

Rigid = rigid_utils.Rigid


class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        all_sample_lengths = range(
            self._samples_cfg.min_length,
            self._samples_cfg.max_length+1,
            self._samples_cfg.length_step
        )
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [
                int(x) for x in samples_cfg.length_subset
            ]
        all_sample_ids = []
        num_batch = self._samples_cfg.num_batch
        assert self._samples_cfg.samples_per_length % num_batch == 0
        self.n_samples = self._samples_cfg.samples_per_length // num_batch

        for length in all_sample_lengths:
            for sample_id in range(self.n_samples):
                sample_ids = torch.tensor([num_batch * sample_id + i for i in range(num_batch)])
                all_sample_ids.append((length, sample_ids))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {
            'num_res': num_res,
            'sample_id': sample_id,
        }
        return batch


def dataset_creation(dataset_class, cfg, task):
    train_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=True,
    ) 
    eval_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=False,
    ) 
    return train_dataset, eval_dataset


def get_available_device(num_device):
    return GPUtil.getAvailable(order='memory', limit = 8)[:num_device]

def run_easy_cluster(designable_dir, output_dir):
    # designable_dir should be a directory with individual PDB files in it that we want to cluster
    # output_dir is where we are going to save the easy cluster output files

    # Returns the number of clusters

    easy_cluster_args = [
        'foldseek',
        'easy-cluster',
        designable_dir,
        os.path.join(output_dir, 'res'),
        output_dir,
        '--alignment-type',
        '1',
        '--cov-mode',
        '0',
        '--min-seq-id',
        '0',
        '--tmscore-threshold',
        '0.5',
    ]
    process = subprocess.Popen(
        easy_cluster_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate()
    del stdout # We don't actually need the stdout, we will read the number of clusters from the output files
    rep_seq_fasta = fasta.FastaFile.read(os.path.join(output_dir, 'res_rep_seq.fasta'))
    return len(rep_seq_fasta)
 


def get_all_top_samples(output_dir, csv_fname='*/*/top_sample.csv'):
    all_csv_paths = glob.glob(os.path.join(output_dir, csv_fname), recursive=True)
    top_sample_csv = pd.concat([pd.read_csv(x) for x in all_csv_paths])
    top_sample_csv.to_csv(
        os.path.join(output_dir, 'all_top_samples.csv'), index=False)
    return top_sample_csv


def calculate_diversity(output_dir, metrics_df, top_sample_csv, designable_csv_path):
    designable_samples = top_sample_csv[top_sample_csv.designable]
    designable_dir = os.path.join(output_dir, 'designable')
    os.makedirs(designable_dir, exist_ok=True)
    designable_txt = os.path.join(designable_dir, 'designable.txt')
    if os.path.exists(designable_txt):
        os.remove(designable_txt)
    with open(designable_txt, 'w') as f:
        for _, row in designable_samples.iterrows():
            sample_path = row.sample_path
            sample_name = f'sample_id_{row.sample_id}_length_{row.length}.pdb'
            write_path = os.path.join(designable_dir, sample_name)
            shutil.copy(sample_path, write_path)
            f.write(write_path+'\n')
    if metrics_df['Total codesignable'].iloc[0] <= 1:
        metrics_df['Clusters'] = metrics_df['Total codesignable'].iloc[0]
    else:
        add_diversity_metrics(designable_dir, metrics_df, designable_csv_path)


def add_diversity_metrics(designable_dir, designable_csv, designable_csv_path):
    designable_txt = os.path.join(designable_dir, 'designable.txt')
    clusters = run_easy_cluster(designable_dir, designable_dir)
    designable_csv['Clusters'] = clusters
    designable_csv.to_csv(designable_csv_path, index=False)


def calculate_pmpnn_consistency(output_dir, designable_csv, designable_csv_path):
    # output dir points to directory containing length_60, length_61, ... etc folders
    sample_dirs = glob.glob(os.path.join(output_dir, 'length_*/sample_*'))
    average_accs = []
    max_accs = []
    for sample_dir in sample_dirs:
        pmpnn_fasta_path = os.path.join(sample_dir, 'self_consistency', 'seqs', 'sample_modified.fasta')
        codesign_fasta_path = os.path.join(sample_dir, 'self_consistency', 'codesign_seqs', 'codesign.fa')
        pmpnn_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
        codesign_fasta = fasta.FastaFile.read(codesign_fasta_path)
        codesign_seq = codesign_fasta['codesign_seq_1']
        accs = []
        for seq in pmpnn_fasta:
            num_matches = sum([1 if pmpnn_fasta[seq][i] == codesign_seq[i] else 0 for i in range(len(pmpnn_fasta[seq]))])
            total_length = len(pmpnn_fasta[seq])
            accs.append(num_matches / total_length)
        average_accs.append(np.mean(accs))
        max_accs.append(np.max(accs))
    designable_csv['Average PMPNN Consistency'] = np.mean(average_accs)
    designable_csv['Average Max PMPNN Consistency'] = np.mean(max_accs)
    designable_csv.to_csv(designable_csv_path, index=False)

def calculate_pmpnn_designability(output_dir, designable_csv, designable_csv_path):
    sample_dirs = glob.glob(os.path.join(output_dir, 'length_*/sample_*'))
    try:
        single_pmpnn_results = []
        top_pmpnn_results = []
        for sample_dir in sample_dirs:
            all_pmpnn_folds_df = pd.read_csv(os.path.join(sample_dir, 'pmpnn_results.csv'))
            single_pmpnn_fold_df = all_pmpnn_folds_df.iloc[[0]]
            single_pmpnn_results.append(single_pmpnn_fold_df)
            min_index = all_pmpnn_folds_df['bb_rmsd'].idxmin()
            top_pmpnn_df = all_pmpnn_folds_df.loc[[min_index]]
            top_pmpnn_results.append(top_pmpnn_df)
        single_pmpnn_results_df = pd.concat(single_pmpnn_results, ignore_index=True)
        top_pmpnn_results_df = pd.concat(top_pmpnn_results, ignore_index=True)
        designable_csv['Single seq PMPNN Designability'] = np.mean(single_pmpnn_results_df['bb_rmsd'].to_numpy() < 2.0)
        designable_csv['Top seq PMPNN Designability'] = np.mean(top_pmpnn_results_df['bb_rmsd'].to_numpy() < 2.0)
        designable_csv.to_csv(designable_csv_path, index=False)
    except:
        # TODO i think it breaks when one process gets here first
        print("calculate pmpnn designability didnt work")



def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def get_ddp_info():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_id = rank // world_size
    return {"node_id": node_id, "local_rank": local_rank, "rank": rank, "world_size": world_size}


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened


def save_traj(
        sample: np.ndarray,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: str,
        aa_traj = None,
        clean_aa_traj = None,
        write_trajectories = True,
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [noisy_T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [clean_T, N, 37, 3] atom37 predictions of clean data at each time step.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.
        aa_traj: [noisy_T, N] amino acids (0 - 20 inclusive).
        clean_aa_traj: [clean_T, N] amino acids (0 - 20 inclusive).
        write_trajectories: bool Whether to also write the trajectories as well
                                 as the final sample

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, 'sample.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj.pdb')
    x0_traj_path = os.path.join(output_dir, 'x0_traj.pdb')

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    noisy_traj_length, num_res, _, _ = bb_prot_traj.shape
    clean_traj_length = x0_traj.shape[0]
    assert sample.shape == (num_res, 37, 3)
    assert bb_prot_traj.shape == (noisy_traj_length, num_res, 37, 3)
    assert x0_traj.shape == (clean_traj_length, num_res, 37, 3)

    if aa_traj is not None:
        assert aa_traj.shape == (noisy_traj_length, num_res)
        assert clean_aa_traj is not None
        assert clean_aa_traj.shape == (clean_traj_length, num_res)

    sample_path = au.write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aa_traj[-1] if aa_traj is not None else None,
    )
    if write_trajectories:
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=aa_traj,
        )
        x0_traj_path = au.write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=clean_aa_traj,
        )
    return {
        'sample_path': sample_path,
        'traj_path': prot_traj_path,
        'x0_traj_path': x0_traj_path,
    }


def get_dataset_cfg(cfg):
    if cfg.data.dataset == 'pdb':
        return cfg.pdb_dataset
    else:
        raise ValueError(f'Unrecognized dataset {cfg.data.dataset}')
