#!/usr/bin/env python3
"""
LLaMA-Factory Training Job Script
This script parses a YAML config file and launches a training job with an adaptive run name.
"""

import os
import sys
import yaml
import socket
import argparse
import subprocess
import time
import uuid
import shutil
from pathlib import Path

def create_pbs_script(config_file, run_name, walltime, ngpus, ncpus, mem, output_dir, job_name):
    """Create a PBS script with the adaptive run name."""
    # Get the current hostname
    hostname = socket.gethostname()
    
    # Append domain if it's not already included
    if "." not in hostname:
        hostname = f"{hostname}.eecscl.psu.edu"
    
    # Create the output_config_path in the output directory
    output_config = os.path.join(output_dir, "training_config.yaml")
    
    # Check if config has GaLore and its fused implementation setting
    galore_mode = ""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            if config.get('use_galore', False):
                fused_enabled = config.get('galore_fused', True)
                galore_mode = f"Using {'fused' if fused_enabled else 'NON-FUSED'} GaLore implementation"
    except Exception:
        # Default message if we can't read the config
        galore_mode = "Using GaLore implementation"
    
    pbs_script = f"""#!/bin/tcsh
#PBS -l ngpus={ngpus}
#PBS -l ncpus={ncpus}
#PBS -l walltime={walltime}
#PBS -q workq@{hostname}
#PBS -N {job_name}
#PBS -M pxm5426@psu.edu 
#PBS -m bea
#PBS -l mem={mem}
#PBS -o pbs_results/
#PBS -e pbs_results/
cd $PBS_O_WORKDIR

# Source the user's tcsh configuration file
if (-e ~/.tcshrc) then
    source ~/.tcshrc
else
    echo "Warning: ~/.tcshrc not found in the home directory!"
endif

# Set the run name for wandb
setenv WANDB_NAME {run_name}
setenv WANDB_PROJECT "llama-factory"
# Enable debug logging for pytorch
setenv TORCH_DISTRIBUTED_DEBUG DETAIL
# Set logging level to INFO
setenv LLAMAFACTORY_VERBOSITY INFO

# # Set FORCE_TORCHRUN environment variable (required for DeepSpeed)
# setenv FORCE_TORCHRUN 1

echo "Starting training run: {run_name}"
echo "Using config file: {output_config}"
echo "Output directory: {output_dir}"

# Activate the llama-factory conda environment
conda activate llama-factory-env

# Check for and install required dependencies
pip install omegaconf --quiet
# Make sure our custom galore implementation is installed from the right location
pip install -e ./galore-torch --quiet

echo "=== Running nvidia-smi to show GPU status ==="
nvidia-smi
echo "=== Starting training with verbose output ==="

# Add note about GaLore implementation
echo "NOTE: Using fixed GaLore implementation that avoids redundant SVD computations"
echo "during gradient accumulation steps."
{f'echo "{galore_mode}"' if galore_mode else ''}

# Monitor GPU usage (TCSH compatible version)
echo "Starting GPU monitoring in background"
nvidia-smi --loop=60 > gpu_monitor.log &
set background_pid=$!

# Run training with the config file in the output directory
llamafactory-cli train {output_config}

# Kill the background process if it exists
if ($?background_pid) then
  echo "Stopping GPU monitoring (PID: $background_pid)"
  kill $background_pid
endif

echo "Training completed or was interrupted."
"""
    return pbs_script, hostname

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
        
        else: # Default case (neither MFSGD nor GaLore, or GaLore without specific LRs)
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

def cleanup_temp_files(temp_script_path):
    """Clean up temporary script files after job submission."""
    try:
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
            print(f"Removed temporary PBS script: {temp_script_path}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}", file=sys.stderr)

def generate_job_id():
    """Generate a unique job ID for the PBS script file name."""
    # Use a timestamp in the format YYYYMMDDHHMMSSmmm (milliseconds)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    # Add a random short UUID to ensure uniqueness
    random_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{random_id}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LLaMA-Factory training job with adaptive run name')
    parser.add_argument('config_file', nargs='?', default='llama3_tulu_sft.yaml', 
                        help='Path to the YAML config file (default: llama3_tulu_sft.yaml)')
    parser.add_argument('--walltime', default='48:00:00', 
                        help='Walltime for the job (default: 48:00:00)')
    parser.add_argument('--ngpus', default='2', 
                        help='Number of GPUs for the job (default: 2)')
    parser.add_argument('--ncpus', default='32', 
                        help='Number of CPUs for the job (default: 32)')
    parser.add_argument('--mem', default='60g', 
                        help='Memory for the job (default: 60g)')
    parser.add_argument('--base_output_dir', default='/scratch/pxm5426/runs/lora-exploration/llama-factory',
                        help='Base output directory where run outputs will be stored (default: /scratch/pxm5426/runs/lora-exploration/llama-factory)')
    parser.add_argument('--keep_temp_files', action='store_true',
                        help='Keep temporary files after job submission (default: False)')
    parser.add_argument('--job_name', default='llama-factory',
                        help='Name of the job for PBS (default: llama-factory)')
    args = parser.parse_args()
    
    # Generate the run name
    run_name = generate_run_name(args.config_file, args.ngpus)
    
    # Create a config file in the output directory
    output_config, output_dir = create_temp_config_with_output_dir(args.config_file, run_name, args.base_output_dir)
    
    # Generate a unique job ID for the PBS script file
    job_id = generate_job_id()
    
    # Create a temporary PBS script
    pbs_script, detected_hostname = create_pbs_script(
        output_config, 
        run_name, 
        args.walltime, 
        args.ngpus, 
        args.ncpus, 
        args.mem,
        output_dir,
        args.job_name
    )
    temp_script_path = f"temp_llama_factory_{job_id}.job"
    
    with open(temp_script_path, 'w') as f:
        f.write(pbs_script)
    
    # Submit the job directly
    print(f"Submitting job '{args.job_name}' (Run: {run_name}) with config '{output_config}'...")
    print(f"Output directory: {output_dir}")
    print(f"Resource configuration: {args.ngpus} GPUs, {args.ncpus} CPUs, {args.mem} memory, {args.walltime} walltime")
    print(f"Target host: {detected_hostname}")
    result = subprocess.run(['qsub', temp_script_path], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip()
        print(f"✓ Job submitted successfully: {job_id} (Run name: {run_name})")
    else:
        print(f"✗ Job submission failed: {result.stderr}")
    
    # Clean up only the temporary script file
    if not args.keep_temp_files:
        cleanup_temp_files(temp_script_path)
    else:
        print(f"Keeping temporary files as requested:")
        print(f"  - PBS script: {temp_script_path}")

if __name__ == "__main__":
    main() 