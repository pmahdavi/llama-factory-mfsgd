import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from llamafactory.hparams import FinetuningArguments
from llamafactory.train.trainer_utils import _create_muon_optimizer
import os

def extract_momentum(checkpoint_dir, model_name_or_path):
    """
    Extracts momentum from a Muon optimizer checkpoint.

    Args:
        checkpoint_dir (str): Path to the checkpoint directory.
        model_name_or_path (str): The name or path of the pre-trained model.

    Returns:
        dict: A dictionary mapping parameter names to their momentum tensors.
    """
    print("Initializing model...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    print("Model initialized.")

    # These arguments are based on the training_config.yaml and the defaults.
    # They are needed to create the optimizer correctly.
    training_args = TrainingArguments(
        output_dir=os.path.dirname(checkpoint_dir),
        learning_rate=5.0e-05,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
    )

    finetuning_args = FinetuningArguments(
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
    )

    print("Creating Muon optimizer...")
    optimizer = _create_muon_optimizer(model, training_args, finetuning_args)
    print("Optimizer created.")

    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    if not os.path.exists(optimizer_path):
        raise FileNotFoundError(f"Optimizer state not found at {optimizer_path}")

    print(f"Loading optimizer state from {optimizer_path}...")
    optimizer_state_dict = torch.load(optimizer_path, map_location="cpu")
    optimizer.load_state_dict(optimizer_state_dict)
    print("Optimizer state loaded.")

    param_map = {id(p): name for name, p in model.named_parameters()}
    
    momentum_dict = {}
    print("Extracting momentum tensors...")
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                if 'momentum_buffer' in optimizer.state[p]:
                    param_name = param_map.get(id(p))
                    if param_name:
                        momentum_tensor = optimizer.state[p]['momentum_buffer']
                        momentum_dict[param_name] = momentum_tensor
                        print(f"  Found momentum for: {param_name} with shape: {momentum_tensor.shape}")

    if not momentum_dict:
        print("No momentum tensors were found. The optimizer might not be Muon or has no momentum state.")
    else:
        main_run_dir = os.path.dirname(checkpoint_dir)
        export_dir = os.path.join(main_run_dir, "export")
        print(f"Creating export directory at: {export_dir}")
        os.makedirs(export_dir, exist_ok=True)
        output_file = os.path.join(export_dir, "momentum_tensors.pt")
        torch.save(momentum_dict, output_file)
        print(f"Momentum tensors saved to {output_file}")

    return momentum_dict

if __name__ == "__main__":
    CHECKPOINT_DIR = "/scratch/pxm5426/runs/lora-exploration/llama-factory/Llama-3.1-8B_tulu3_mixture_math_reasoning_full_muon_ebs128_lr5e-05/checkpoint-2611"
    MODEL_NAME_OR_PATH = "meta-llama/Llama-3.1-8B"
    
    extract_momentum(CHECKPOINT_DIR, MODEL_NAME_OR_PATH) 