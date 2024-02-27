# Multiflow: protein co-design with discrete and continuous flows

Multiflow is a protein sequence and structure generative model based on our preprint: [Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997). 

Our codebase is developed on top of [FrameFlow](https://github.com/microsoft/frame-flow).
The sequence generative model is adpated from [Discrete Flow Models (DFM)](https://github.com/andrew-cr/discrete_flow_models).

If you use this codebase, then please cite

```
@article{campbell2024generative,
  title={Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design},
  author={Campbell, Andrew and Yim, Jason and Barzilay, Regina and Rainforth, Tom and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2402.04997},
  year={2024}
}
```

> [!NOTE]  
> This codebase is very fresh. We expect there to be bugs and issues with other systems and environments.
> Please create a github issue or pull request and we will attempt to help.

LICENSE: MIT

![multiflow-landing-page](https://github.com/jasonkyuyim/multiflow/blob/main/media/codesign.gif)

## Installation

We recommend using [mamba](https://mamba.readthedocs.io/en/latest/).
If using mamba then use `mamba` in place of `conda`.

```bash
# Install environment with dependencies.
conda env create -f multiflow.yml

# Activate environment
conda activate multiflow

# Install local package.
# Current directory should have setup.py.
pip install -e .
```

Next you need to install torch-scatter manually depending on your torch version.
(Unfortunately torch-scatter has some oddity that it can't be installed with the environment.)
We use torch 2.0.1 and cuda 11.7 so we install the following

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

If you use a different torch then that can be found with the following.
```bash
# Find your installed version of torch
python
>>> import torch
>>> torch.__version__
# Example: torch 2.0.1+cu117
```

> [!WARNING]  
> You will likely run into the follow error from DeepSpeed
```bash
ModuleNotFoundError: No module named 'torch._six'
```
If so, replace `from torch._six import inf` with `from torch import inf`.
* `/path/to/envs/site-packages/deepspeed/runtime/utils.py`
* `/path/to/envs/site-packages/deepspeed/runtime/zero/stage_1_and_2.py`

where `/path/to/envs` is replaced with your path. We would appreciate a pull request to avoid this monkey patch!

## Wandb

Our training relies on logging with wandb. Log in to Wandb and make an account.
Authorize Wandb [here](https://wandb.ai/authorize).

## Data

We host the datasets on Zenodo [here](https://zenodo.org/records/10714631?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjMTk2YjlmLTM4OTUtNGVhYi1hODcxLWE1ZjExOTczY2IzZiIsImRhdGEiOnt9LCJyYW5kb20iOiI4MDY5ZDUzYjVjMTNhNDllMDYxNmI3Yjc2NjcwYjYxZiJ9.C2eZZmRu-nu7H330G-DkV5kttfjYB3ANozdOMNm19uPahvtLrDRvd_4Eqlyb7lp24m06e4OHhHQ4zlj68S1O_A).
Download the following files,
* `real_train_set.tar.gz` (2.5 GB)
* `synthetic_train_set.tar.gz` (220 MB)
* `test_set.tar.gz` (347 MB)
Next, untar the files
```bash
# Uncompress training data
mkdir train_set
tar -xzvf real_train_set.tar.gz -C train_set/
tar -xzvf synthetic_train_set.tar.gz -C train_set/

# Uncompress test data
mkdir test_set
tar -xzvf test_set.tar.gz -C test_set/
```
The resulting directory structure should look like
```bash
<current_dir>
├── train_set
│   ├── processed_pdb
|   |   ├── <subdir>
|   |   |   └── <protein_id>.pkl
│   ├── processed_synthetic
|   |   └── <protein_id>.pkl
├── test_set
|   └── processed
|   |   ├── <subdir>
|   |   |   └── <protein_id>.pkl
...
```
Our experiments read the data by using relative paths. Keep the directory structure like this to avoid bugs.

## Training

The command to run co-design training is the following, 
```bash
python -W ignore multiflow/experiments/train_se3_flows.py -cn pdb_codesign
```
We use [Hydra](https://hydra.cc/) to maintain our configs. 
The training config is found here `multiflow/configs/pdb_codesign.yaml`.

Most important fields:
* `experiment.num_devices`: Number of GPUs to use for training. Default is 2.
* `data.sampler.max_batch_size`: Maximum batch size. We use dynamic batch sizes depending on `data.sampler.max_num_res_squared`. Both these parameters need to be tuned for your GPU memory. Our default settings are set for a 40GB Nvidia RTX card.
* `data.sampler.max_num_res_squared`: See above.


## Inference

We provide pre-trained model weights at this [Zenodo link](https://zenodo.org/records/10714631?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjMTk2YjlmLTM4OTUtNGVhYi1hODcxLWE1ZjExOTczY2IzZiIsImRhdGEiOnt9LCJyYW5kb20iOiI4MDY5ZDUzYjVjMTNhNDllMDYxNmI3Yjc2NjcwYjYxZiJ9.C2eZZmRu-nu7H330G-DkV5kttfjYB3ANozdOMNm19uPahvtLrDRvd_4Eqlyb7lp24m06e4OHhHQ4zlj68S1O_A).

Run the following to unpack the weights
```bash
tar -xzvf weights.tar.gz
```

The following three tasks can be performed. 
```bash
# Unconditional Co-Design
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional

# Inverse Folding
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_inverse_folding

# Forward Folding
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_forward_folding
```

### Configs

Config locations:
- configs/inference_unconditional.yaml: unconditional sampling config.
- configs/inference_inverse_folding.yaml: inverse folding config.
- configs/inference_forward_folding.yaml: forward folding config.

Most important fields:
- inference.num_gpus: Number of GPUs to use. I typically use 2 or 4.

- inference.{...}_ckpt_path: Checkpoint path for hallucination.

- inference.interpolant.sampling.num_timesteps: Number of steps in the flow.

- inference.folding.folding_model: `esmf` for ESMFold and `af2` for AlphaFold2.

[Only for hallucination]
- inference.samples.samples_per_length: Number of samples per length.
- inference.samples.min_length: Start of length range to sample.
- inference.samples.max_length: End of length range to sample.
- inference.samples.length_subset: Subset of lengths to sample. Will override min_length and max_length.