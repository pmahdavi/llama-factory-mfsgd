# MoFaSGD: Low-rank Momentum Factorization for Memory Efficient Training

This repository contains the official implementation of the paper **"Low-rank Momentum Factorization for Memory Efficient Training"** ([TMLR, 2025](https://openreview.net/forum?id=W3D3TVo9a3)). We introduce MoFaSGD, a memory-efficient optimizer that maintains a dynamically updated low-rank SVD representation of the first-order momentum, closely approximating its full-rank counterpart throughout training. This factorization enables a memory-efficient fine-tuning method that adaptively updates the optimization subspace at each iteration.

Our work demonstrates that MoFaSGD achieves a competitive trade-off between memory reduction and performance compared to state-of-the-art low-rank optimization methods like LoRA and GaLore, as well as full-parameter fine-tuning with AdamW.

## Acknowledgement

This work is built upon the excellent [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository. We thank the authors for their wonderful work and for making their code publicly available.

## Features

- **Memory-Efficient Training**: Fine-tune large language models with significantly less memory, comparable to LoRA.
- **Dynamic Subspace Updates**: Adaptively updates the optimization subspace at each iteration for better performance.
- **Full-Parameter Updates**: Enables full-parameter updates while operating in a lower-dimensional space.

## Installation

1.  Clone the repository and initialize the submodules:
    ```bash
    git clone --recursive https://github.com/pmahdavi/llama-factory-mfsgd.git
    cd llama-factory-mfsgd
    ```
    If you have already cloned the repository without the `--recursive` flag, you can initialize the submodules by running:
    ```bash
    git submodule update --init --recursive
    ```

2.  Create and activate a conda environment:
    ```bash
    conda create --name llama-factory-env python=3.10
    conda activate llama-factory-env
    ```

3.  Install the required dependencies:
    ```bash
    pip install -e ".[torch,metrics]"
    ```
    This will install all the necessary packages, including PyTorch and other core libraries. For more details on optional dependencies, please refer to the original [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository.

4. Install the custom GaLore implementation:
    ```bash
    pip install -e galore-torch/
    ```

## Usage

The `run.py` script is the main entry point for running experiments.

To run an experiment:
```bash
python run.py <path_to_config_file> [options]
```

For example, to fine-tune a model with MoFaSGD on your local machine:

```bash
python run.py configs/mfsgd/llama3.1_8b_sft_mfsgd_lr.yaml --ngpus 1
```

### Available Arguments

-   `config_file`: Path to the YAML config file.
-   `--ngpus`: Number of GPUs for the job (default: `2`).
-   `--base_output_dir`: Base output directory for runs.

## Experiment Setup

This repository contains the code and configurations for the **LLaMA-3.1 8B instruction-tuning experiments on the Tulu3 dataset**, as described in the MoFaSGD paper.

### Example Configurations

Example configurations for the experiments can be found in the `configs/mfsgd/` directory. These files provide a starting point for running experiments with different optimizers:

-   **MoFaSGD**: `configs/mfsgd/llama3.1_8b_sft_mfsgd_lr.yaml`
-   **GaLore**: `configs/mfsgd/llama3.1_8b_sft_galore.yaml`
-   **LoRA**: `configs/mfsgd/llama3.1_8b_sft_lora.yaml`
-   **AdamW**: `configs/mfsgd/llama3.1_8b_sft_adamw_bf16.yaml`

You can modify these files or create new ones to run your own experiments.

### Memory Profiling

To facilitate the memory analysis presented in the paper, we have integrated a memory profiling tool based on the PyTorch memory profiler. To use it, add the following parameters to your configuration YAML file:

```yaml
profile_memory_from_start: true
profile_memory_stop_step: 4
profile_memory_max_entries: 10000000
```

-   `profile_memory_from_start`: Set to `true` to begin profiling from the start of the training.
-   `profile_memory_stop_step`: The profiler will dump a snapshot of the memory usage after this many optimizer steps and then stop.
-   `profile_memory_max_entries`: The maximum number of memory allocation/deallocation events to record.

Example command:
```bash
python run.py <your_profiling_config.yaml> --ngpus 1
```

## Citation

If you find our work useful, please cite our paper:

```bibtex
@article{mahdavinia2025mofasgd,
  title={Low-rank Momentum Factorization for Memory Efficient Training},
  author={Mahdavinia, Pouria and Mahdavi, Mehrdad},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://openreview.net/forum?id=W3D3TVo9a3}
}
``` 