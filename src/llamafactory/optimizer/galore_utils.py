import torch
from ..train.trainer_utils import logger # Assuming logger is accessible here

def print_galore_parameter_status(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Print all trainable parameters along with their GaLore status.

    Args:
        model: The model with parameters.
        optimizer: The GaLore optimizer (e.g., GaLoreAdamW).
    """
    logger.info_rank0("\n" + "="*80)
    logger.info_rank0("TRAINABLE PARAMETERS WITH GALORE STATUS")
    logger.info_rank0("="*80)

    galore_param_ids = set()
    non_galore_param_ids = set()

    for group in optimizer.param_groups:
        is_galore_group = "rank" in group  # GaLore groups have a 'rank' key
        for p in group["params"]:
            if p.requires_grad:
                if is_galore_group:
                    galore_param_ids.add(id(p))
                else:
                    non_galore_param_ids.add(id(p))

    total_params = 0
    galore_params_count = 0
    non_galore_params_count = 0

    # Determine the maximum name length for formatting
    max_name_len = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(name) > max_name_len:
                max_name_len = len(name)
    max_name_len = max(max_name_len, len("Parameter Name")) # Ensure header fits

    logger.info_rank0(f"{ 'Parameter Name':<{max_name_len+5}} {'Shape':<20} {'Num Params':<15} {'GaLore?':<10}")
    logger.info_rank0("-" * (max_name_len + 5 + 20 + 15 + 10))

    for name, param in model.named_parameters():
        if param.requires_grad:
            is_galore = id(param) in galore_param_ids
            # It's possible a param is in neither if it's trainable but somehow not in any optimizer group,
            # though this shouldn't happen with the current setup.
            # We prioritize checking galore_param_ids first.
            
            num_params = param.numel()
            total_params += num_params

            if is_galore:
                galore_params_count += num_params
            else: # If not in galore_param_ids, and it's trainable, it must be in non_galore_param_ids or an issue
                non_galore_params_count += num_params

            logger.info_rank0(f"{name:<{max_name_len+5}} {str(list(param.shape)):<20} {num_params:<15,d} {'✓' if is_galore else '✗'}")

    logger.info_rank0("-" * (max_name_len + 5 + 20 + 15 + 10))
    logger.info_rank0(f"Total trainable parameters: {total_params:,}")
    if total_params > 0:
        logger.info_rank0(f"GaLore parameters:          {galore_params_count:,} ({galore_params_count/total_params*100:.2f}%)")
        logger.info_rank0(f"Non-GaLore parameters:      {non_galore_params_count:,} ({non_galore_params_count/total_params*100:.2f}%)")
    logger.info_rank0("="*80 + "\n") 