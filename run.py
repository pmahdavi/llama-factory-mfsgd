#!/usr/bin/env python3
"""
LLaMA-Factory Training Job Script
This script parses a YAML config file and launches a training job with an adaptive run name.
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path

def generate_run_name(config_file, ngpus=4):
    """Generate an adaptive run name based on the config file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract model name
        model_name = Path(config.get('model_name_or_path', config.get('model_name', "unknown_model"))).name
            
        # Extract dataset name
        dataset_value = config.get('dataset_name', config.get('dataset', config.get('datasets', "unknown_dataset")))
        if isinstance(dataset_value, list):
            dataset = '_'.join(dataset_value)
        else:
            dataset = str(dataset_value) # Ensure dataset is a string
            
        finetuning_type = config.get('finetuning_type', '')
        
        bs_per_device = config.get('per_device_train_batch_size', 0)
        gradient_accumulation = config.get('gradient_accumulation_steps', 1)
        num_gpus = int(ngpus)
        effective_bs = bs_per_device * gradient_accumulation * num_gpus if bs_per_device else 0
        
        components = [model_name, dataset]
        if finetuning_type:
            components.append(finetuning_type)
        
        lr_component = ""
        optimizer_specific_tags = []

        if config.get('use_mfsgd', False):
            mfsgd_tag_parts = ["mfsgd"]
            rank = config.get('mfsgd_rank')
            if rank is not None:
                mfsgd_tag_parts.append(f'r{rank}')
            
            eps = config.get('mfsgd_eps')
            if eps is not None:
                mfsgd_tag_parts.append(f'eps{eps:.0e}'.replace('e-0', 'e-')) # Format like 1e-8

            if config.get('mfsgd_use_ones_for_nonzero_s', False):
                mfsgd_tag_parts.append('s1')
            
            max_val = config.get('mfsgd_max_value')
            if max_val is not None:
                mfsgd_tag_parts.append(f'max{max_val:.0e}'.replace('+0','')) # Format like 1e4

            optimizer_specific_tags.append("-".join(mfsgd_tag_parts))

            lr_mfsgd = config.get('learning_rate_mfsgd')
            lr_adamw = config.get('learning_rate_adamw')
            mfsgd_lr_str_parts = []
            if lr_mfsgd is not None:
                mfsgd_lr_str_parts.append(f'mfLR{lr_mfsgd}')
            if lr_adamw is not None:
                mfsgd_lr_str_parts.append(f'admLR{lr_adamw}')
            
            if mfsgd_lr_str_parts:
                lr_component = '_'.join(mfsgd_lr_str_parts)
            elif config.get('learning_rate') is not None:
                lr_component = f"lr{config.get('learning_rate')}"

        elif config.get('use_galore', False):
            galore_tag_parts = []
            if config.get('galore_fused', True):
                galore_tag_parts.append('galoreF') # Shorter: F for fused
            else:
                galore_tag_parts.append('galore')
            
            galore_rank = config.get('galore_rank')
            if galore_rank is not None:
                galore_tag_parts.append(f'r{galore_rank}')
            optimizer_specific_tags.append("-".join(galore_tag_parts))
            
            # GaLore specific LRs
            lr_galore = config.get('galore_lr_galore_params')
            lr_non_galore = config.get('galore_lr_non_galore_params')
            galore_lr_str_parts = []
            if lr_galore is not None:
                galore_lr_str_parts.append(f'glr{lr_galore}') # e.g., glr1e-4
            if lr_non_galore is not None:
                galore_lr_str_parts.append(f'non_glr{lr_non_galore}') # e.g., non_glr5e-6
            
            if galore_lr_str_parts:
                lr_component = '_'.join(galore_lr_str_parts)
            elif config.get('learning_rate') is not None: # Fallback to global LR if specific GaLore LRs are not set
                 lr_component = f"lr{config.get('learning_rate')}"
        
        elif config.get('use_muon', False): # Added Muon configuration handling
            muon_tag_parts = ["muon"]
            
            # Add momentum if not default (0.95) or if explicitly set (e.g., 0.0)
            muon_momentum = config.get('muon_momentum')
            if muon_momentum is not None:
                if muon_momentum == 0.0: # Handle explicit 0.0
                    muon_tag_parts.append('mom0')
                elif muon_momentum != 0.95:
                    muon_tag_parts.append(f'mom{muon_momentum}')
            
            # Add nesterov if False (default is True)
            muon_nesterov = config.get('muon_nesterov')
            if muon_nesterov is False: # Check for explicit False
                muon_tag_parts.append('noNest')
            
            # Add ns_steps if not default (5)
            muon_ns_steps = config.get('muon_ns_steps')
            if muon_ns_steps is not None and muon_ns_steps != 5:
                muon_tag_parts.append(f'ns{muon_ns_steps}')
            
            optimizer_specific_tags.append("-".join(muon_tag_parts))
            
            # Muon uses the global learning_rate
            if config.get('learning_rate') is not None:
                lr_component = f"lr{config.get('learning_rate')}"

        else: # Default case (neither MFSGD, GaLore, nor Muon)
            # Get the standard optimizer name, default to adamw_torch if not specified
            optimizer_name = config.get('optim', 'adamw_torch').replace('_torch', '') # e.g. adamw_torch -> adamw
            optimizer_specific_tags.append(optimizer_name)
            
            if config.get('learning_rate') is not None:
                lr_component = f"lr{config.get('learning_rate')}"

        if optimizer_specific_tags:
            components.extend(optimizer_specific_tags)
        
        if effective_bs:
            components.append(f'ebs{effective_bs}')
        
        if lr_component:
            components.append(lr_component)
        
        run_name = '_'.join([str(c) for c in components if c and str(c)]) # Ensure c is not empty string before join
        return run_name
    except Exception as e:
        print(f"Warning: Could not parse config file for run name: {e}", file=sys.stderr)
        return f'llama_factory_run_{os.path.basename(config_file).split(".")[0]}'

def create_temp_config_with_output_dir(config_file, run_name, base_output_dir):
    """Create a temporary config file with updated output_dir."""
    # Read the original config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build dynamic output directory
    output_dir = os.path.join(base_output_dir, run_name)
    
    # Update the config
    config['output_dir'] = output_dir
    
    # Also update the run_name in the config if it exists (for wandb)
    if 'run_name' in config:
        config['run_name'] = run_name
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the configuration directly to the output directory
    output_config = os.path.join(output_dir, "training_config.yaml")
    with open(output_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created configuration file at: {output_config}")
    return output_config, output_dir

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LLaMA-Factory training job with adaptive run name')
    parser.add_argument('config_file', nargs='?', default='llama3_tulu_sft.yaml', 
                        help='Path to the YAML config file (default: llama3_tulu_sft.yaml)')
    parser.add_argument('--ngpus', default='2', 
                        help='Number of GPUs for the job (default: 2)')
    parser.add_argument('--base_output_dir', default='/scratch/pxm5426/runs/lora-exploration/llama-factory',
                        help='Base output directory where run outputs will be stored (default: /scratch/pxm5426/runs/lora-exploration/llama-factory)')
    parser.add_argument('--enable_mfsgd_debug_empty_cache', action='store_true',
                        help='Enable MFSGD_DEBUG_EMPTY_CACHE for cache clearing in MFSGD hook (for debug profiling).')
    args = parser.parse_args()
    
    # Generate the run name
    run_name = generate_run_name(args.config_file, args.ngpus)
    
    # Create a config file in the output directory
    output_config, output_dir = create_temp_config_with_output_dir(args.config_file, run_name, args.base_output_dir)

    # Run the training job directly
    print(f"Running job locally (Run: {run_name}) with config '{output_config}'...")
    print(f"Output directory: {output_dir}")
    print(f"GPU configuration: {args.ngpus} GPUs")

    # Set environment variables for local run
    os.environ['WANDB_NAME'] = run_name
    os.environ['WANDB_PROJECT'] = "llama-factory"
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['LLAMAFACTORY_VERBOSITY'] = 'INFO'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if args.enable_mfsgd_debug_empty_cache:
        os.environ['MFSGD_DEBUG_EMPTY_CACHE'] = '1'

    try:
        # Construct the training command
        train_command = f"source ~/.tcshrc; conda activate llama-factory-env; llamafactory-cli train {output_config}"
        
        # Execute the training command
        process = subprocess.Popen(train_command, shell=True, executable='/bin/tcsh')
        process.wait()
        
        if process.returncode == 0:
            print(f"✓ Local training completed successfully (Run name: {run_name})")
        else:
            print(f"✗ Local training failed with return code {process.returncode}")

    except Exception as e:
        print(f"✗ An error occurred during local training: {e}")

if __name__ == "__main__":
    main() 