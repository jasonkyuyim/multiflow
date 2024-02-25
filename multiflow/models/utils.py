import math
import torch
import re
import numpy as np
import pandas as pd
import mdtraj as md
from torch.nn import functional as F
from multiflow.data import utils as du
from openfold.utils.superimposition import superimpose
from openfold.np import residue_constants


CA_IDX = residue_constants.atom_order['CA'] 

def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def sinusoidal_encoding(v, N, D):
	"""Taken from GENIE.
	
	Args:

	"""
	# v: [*]

	# [D]
	k = torch.arange(1, D+1).to(v.device)

	# [*, D]
	sin_div_term = N ** (2 * k / D)
	sin_div_term = sin_div_term.view(*((1, ) * len(v.shape) + (len(sin_div_term), )))
	sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)

	# [*, D]
	cos_div_term = N ** (2 * (k - 1) / D)
	cos_div_term = cos_div_term.view(*((1, ) * len(v.shape) + (len(cos_div_term), )))
	cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)

	# [*, D]
	enc = torch.zeros_like(sin_enc).to(v.device)
	enc[..., 0::2] = cos_enc[..., 0::2]
	enc[..., 1::2] = sin_enc[..., 1::2]

	return enc.to(v.dtype)


def distance(p, eps=1e-10):
    # [*, 2, 3]
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5


def dist_from_ca(trans):

	# [b, n_res, n_res, 1]
	d = distance(torch.stack([
		trans.unsqueeze(2).repeat(1, 1, trans.shape[1], 1), # Ca_1
		trans.unsqueeze(1).repeat(1, trans.shape[1], 1, 1), # Ca_2
	], dim=-2)).unsqueeze(-1)

	return d


def calc_rbf(ca_dists, num_rbf, D_min=1e-3, D_max=22.):
    # Distance radial basis function
    device = ca_dists.device
    D_mu = torch.linspace(D_min, D_max, num_rbf).to(device)
    D_mu = D_mu.view([1,1,1,-1])
    D_sigma = (D_max - D_min) / num_rbf
    return torch.exp(-((ca_dists - D_mu) / D_sigma)**2)


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses



def process_folded_outputs(sample_path, folded_output, true_bb_pos=None):
    mpnn_results = {
        'header': [],
        'sequence': [],
        'bb_rmsd': [],
        'mean_plddt': [],
        'folded_path': [],
    }

    if true_bb_pos is not None:
        mpnn_results['bb_rmsd_to_gt'] = []
        mpnn_results['fold_model_bb_rmsd_to_gt'] = []

    sample_feats = du.parse_pdb_feats('sample', sample_path)
    sample_ca_pos = sample_feats['bb_positions']
    def _calc_ca_rmsd(mask, folded_ca_pos):
        return superimpose(
            torch.tensor(sample_ca_pos)[None],
            torch.tensor(folded_ca_pos[None]),
            mask
        )[1].rmsd[0].item()

    sample_bb_pos = sample_feats['atom_positions'][:, :3].reshape(-1, 3)
    def _calc_bb_rmsd(mask, sample_bb_pos, folded_bb_pos):
        aligned_rmsd = superimpose(
            torch.tensor(sample_bb_pos)[None],
            torch.tensor(folded_bb_pos[None]),
            mask[:, None].repeat(1, 3).reshape(-1)
        )
        return aligned_rmsd[1].item()

    for _, row in folded_output.iterrows():
        folded_feats = du.parse_pdb_feats('folded', row.folded_path)
        seq = du.aatype_to_seq(folded_feats['aatype'])
        folded_ca_pos = folded_feats['bb_positions']
        folded_bb_pos = folded_feats['atom_positions'][:, :3].reshape(-1, 3)

        res_mask = torch.ones(folded_ca_pos.shape[0])

        if true_bb_pos is not None:
            bb_rmsd_to_gt = _calc_bb_rmsd(res_mask, sample_bb_pos, true_bb_pos)
            mpnn_results['bb_rmsd_to_gt'].append(bb_rmsd_to_gt)
            fold_model_bb_rmsd_to_gt = _calc_bb_rmsd(res_mask, folded_bb_pos, true_bb_pos)
            mpnn_results['fold_model_bb_rmsd_to_gt'].append(fold_model_bb_rmsd_to_gt)
        bb_rmsd = _calc_bb_rmsd(res_mask, sample_bb_pos, folded_bb_pos)
        mpnn_results['bb_rmsd'].append(bb_rmsd)
        mpnn_results['folded_path'].append(row.folded_path)
        mpnn_results['header'].append(row.header)
        mpnn_results['sequence'].append(seq)
        mpnn_results['mean_plddt'].append(row.plddt)
    mpnn_results = pd.DataFrame(mpnn_results)
    mpnn_results['sample_path'] = sample_path
    return mpnn_results

def extract_clusters_from_maxcluster_out(file_path):
    # Extracts cluster information from the stdout of a maxcluster run
    cluster_to_paths = {}
    paths_to_cluster = {}
    read_mode = False
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line == "INFO  : Item     Cluster\n":
                read_mode = True
                continue

            if line == "INFO  : ======================================\n":
                read_mode = False

            if read_mode:
                # Define a regex pattern to match the second number and the path
                pattern = r"INFO\s+:\s+\d+\s:\s+(\d+)\s+(\S+)"

                # Use re.search to find the first match in the string
                match = re.search(pattern, line)

                # Check if a match is found
                if match:
                    # Extract the second number and the path
                    cluster_id = match.group(1)
                    path = match.group(2)
                    if cluster_id not in cluster_to_paths:
                        cluster_to_paths[cluster_id] = [path]
                    else:
                        cluster_to_paths[cluster_id].append(path)
                    paths_to_cluster[path] = cluster_id

                else:
                    raise ValueError(f"Could not parse line: {line}")

    return cluster_to_paths, paths_to_cluster

def calc_mdtraj_metrics(pdb_path):
    try:
        traj = md.load(pdb_path)
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_coil_percent = np.mean(pdb_ss == 'C')
        pdb_helix_percent = np.mean(pdb_ss == 'H')
        pdb_strand_percent = np.mean(pdb_ss == 'E')
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent 
        pdb_rg = md.compute_rg(traj)[0]
    except IndexError as e:
        print('Error in calc_mdtraj_metrics: {}'.format(e))
        pdb_ss_percent = 0.0
        pdb_coil_percent = 0.0
        pdb_helix_percent = 0.0
        pdb_strand_percent = 0.0
        pdb_rg = 0.0
    return {
        'non_coil_percent': pdb_ss_percent,
        'coil_percent': pdb_coil_percent,
        'helix_percent': pdb_helix_percent,
        'strand_percent': pdb_strand_percent,
        'radius_of_gyration': pdb_rg,
    }

def calc_aatype_metrics(generated_aatypes):
    # generated_aatypes (B, N)
    unique_aatypes, raw_counts = np.unique(generated_aatypes, return_counts=True)

    # pad with 0's in case it didn't generate any of a certain type
    clean_counts = []
    for i in range(20):
        if i in unique_aatypes:
            clean_counts.append(raw_counts[np.where(unique_aatypes == i)[0][0]])
        else:
            clean_counts.append(0)

    # from the scope128 dataset
    reference_normalized_counts = [
        0.0739, 0.05378621, 0.0410424, 0.05732177, 0.01418736, 0.03995128,
        0.07562267, 0.06695857, 0.02163064, 0.0580802, 0.09333149, 0.06777057,
        0.02034217, 0.03673995, 0.04428474, 0.05987899, 0.05502958, 0.01228988,
        0.03233601, 0.07551553
    ]

    reference_normalized_counts = np.array(reference_normalized_counts)

    normalized_counts = clean_counts / np.sum(clean_counts)

    # compute the hellinger distance between the normalized counts
    # and the reference normalized counts

    hellinger_distance = np.sqrt(np.sum(np.square(np.sqrt(normalized_counts) - np.sqrt(reference_normalized_counts))))

    return {
        'aatype_histogram_dist': hellinger_distance
    }

def calc_ca_ca_metrics(ca_pos, bond_tol=0.1, clash_tol=1.0):
    ca_bond_dists = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca))
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + bond_tol))
    
    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
    clashes = inter_dists < clash_tol
    return {
        'ca_ca_deviation': ca_ca_dev,
        'ca_ca_valid_percent': ca_ca_valid,
        'num_ca_ca_clashes': np.sum(clashes),
    }