import torch
from torch.optim import Optimizer

#############################################
#    MomentumFactor helper class
#############################################
class MomentumFactor:
    """
    Stores a rank-r factorization (U, S, V) for the momentum matrix:
      M ≈ U diag(S) V^T,  with U in R^(m x r), S in R^r, V in R^(n x r).
    Initialization uses a truncated SVD of the parameter p.
    """
    def __init__(self, p: torch.Tensor, rank: int):
        """
        Args:
          p:    (m x n) parameter tensor
          rank: desired rank r
        """
        m, n = p.shape
        with torch.no_grad():
            U_full, S_full, V_full = torch.linalg.svd(p.data)
        r_trunc = min(rank, min(m, n))
        self.U = U_full[:, :r_trunc].clone()
        self.S = S_full[:r_trunc].clone()
        self.V = V_full[:, :r_trunc].clone()


#############################################
#   The MomentumFactorizedSGD Optimizer
#############################################
class MomentumFactorizedSGD(Optimizer):
    r"""
    Implements a rank-r factorized momentum update using backward hooks.
    The full gradient is projected onto low-rank factors and then zeroed out to save memory.

    Key aspects:
    1. Factors U, S, V for momentum M_t ≈ U_t diag(S_t) V_t^T are stored.
    2. Backward hooks on 2D parameters in MFSGD groups compute projections:
          GV   = sum(grad @ V)
          GTU  = sum(grad.t() @ U)
          UTGV = sum(U.t() @ grad @ V)
       The original `grad` tensor is zeroed in-place after projections.
    3. The `step()` method uses these accumulated projections (GV, GTU, UTGV)
       to update U, S, V via a block-matrix SVD approach.
    4. The parameter update `p.data` includes a term from the updated low-rank
       factors (eta1) and an optional orthogonal complement term (eta2).
       **Note**: Since the full gradient is zeroed by the hook, the orthogonal
       complement term will be zero unless G_t is reconstructed/approximated.
       Currently, it is calculated based on a zero gradient if eta2 > 0.

    Args:
      params: iterable of parameters to optimize
      lr:     global learning rate
      rank:   integer rank r for MFSGD factorization
      beta:   momentum decay factor (0.0 <= beta < 1.0)
      eta1:   scale factor for the low-rank momentum term in parameter update
      eta2:   scale factor for the orthogonal complement gradient term in parameter update
      use_current_projection: For orthogonal complement, use U_next, V_next (True) or U, V (False).
                              Currently has no effect if G_t for complement is zero.
      use_ones_for_nonzero_s: If True, use 1.0 for reciprocal of non-zero singular values, else use 1/S.
      mfsgd_eps: Epsilon for thresholding singular values (S_next.abs() > mfsgd_eps).
      nesterov: If True, applies Nesterov momentum (current hook-based version is simplified).
      max_value: Maximum clamp value for reciprocal singular values.
      adam_betas: Tuple (beta1, beta2) for AdamW fallback.
      adam_eps: Epsilon for AdamW fallback.
      adam_weight_decay: Weight decay for AdamW fallback.
    """

    def __init__(self, params,\
                 lr: float = 1e-2,\
                 rank: int = 2,\
                 beta: float = 0.9,\
                 eta1: float = 1.0,\
                 eta2: float = 1.0,\
                 use_current_projection: bool = False,\
                 use_ones_for_nonzero_s: bool = False,\
                 mfsgd_eps: float = 1e-4,\
                 nesterov: bool = False,\
                 max_value: float = 10000,\
                 adam_betas: tuple = (0.9, 0.999),\
                 adam_eps: float = 1e-8,\
                 adam_weight_decay: float = 0.0):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta (MFSGD): {}".format(beta))
        if not 0.0 <= adam_eps:
            raise ValueError("Invalid adam_eps: {}".format(adam_eps))
        if not 0.0 <= adam_weight_decay:
            raise ValueError("Invalid adam_weight_decay: {}".format(adam_weight_decay))

        defaults = dict(lr=lr, rank=rank, beta=beta, eta1=eta1, eta2=eta2,
                        use_current_projection=use_current_projection,
                        use_ones_for_nonzero_s=use_ones_for_nonzero_s,
                        mfsgd_eps=mfsgd_eps,
                        nesterov=nesterov,
                        max_value=max_value,
                        adam_betas=adam_betas,
                        adam_eps=adam_eps,
                        adam_weight_decay=adam_weight_decay)

        super().__init__(params, defaults)

        for group in self.param_groups:
            if group.get('is_mfsgd_group', False):
                group_rank = group['rank']
                for p in group['params']:
                    if p.dim() == 2 and p.requires_grad:
                        state = self.state[p]
                        mf_init = MomentumFactor(p, group_rank)
                        state['U'] = mf_init.U.clone().detach()
                        state['S'] = mf_init.S.clone().detach()
                        state['V'] = mf_init.V.clone().detach()
                        p.register_hook(self._make_backward_hook(p))

    def _make_backward_hook(self, p):
        def hook(grad):
            if grad is None:
                return grad

            state = self.state[p]
            U = state['U']
            V = state['V']
            
            gv = torch.matmul(grad, V)
            gtu = torch.matmul(grad.t(), U)
            utgv = torch.matmul(U.t(), torch.matmul(grad, V))

            for key, proj_tensor in [('GV', gv), ('GTU', gtu), ('UTGV', utgv)]:
                if key not in state:
                    state[key] = torch.zeros_like(proj_tensor)
                state[key].add_(proj_tensor)
            
            grad.zero_()
            return grad
        return hook

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group['lr']
            rank = group['rank']
            beta = group['beta']
            eta1 = group['eta1']
            eta2 = group['eta2']
            use_current_projection = group['use_current_projection']
            use_ones_for_nonzero_s = group['use_ones_for_nonzero_s']
            mfsgd_eps_val = group['mfsgd_eps']
            nesterov = group['nesterov']
            max_value = group['max_value']

            adam_beta1, adam_beta2 = group['adam_betas']
            adam_eps_val = group['adam_eps']
            adam_weight_decay = group['adam_weight_decay']

            apply_mfsgd_logic = group.get('is_mfsgd_group', False)

            for p in group['params']:
                state = self.state[p]

                if apply_mfsgd_logic and p.dim() == 2:
                    U = state['U']
                    S = state['S']
                    V = state['V']

                    m, _ = U.shape # r_dim_u not used, rank from S
                    n, _ = V.shape # r_dim_v not used, rank from S
                    r = S.shape[0]

                    GV = state.pop('GV', torch.zeros(m, r, device=U.device, dtype=U.dtype))
                    GTU = state.pop('GTU', torch.zeros(n, r, device=V.device, dtype=V.dtype))
                    UTGV = state.pop('UTGV', torch.zeros(r, r, device=U.device, dtype=U.dtype))

                    if nesterov:
                        # Nesterov logic with hooks is complex and currently simplified.
                        # True Nesterov would require grad at a future point, problematic with immediate projection.
                        pass

                    U_next, S_next, V_next = self._update_momentum_factor_from_projections(
                        U, S, V, GV, GTU, UTGV, beta, rank
                    )

                    state['U'] = U_next
                    state['S'] = S_next
                    state['V'] = V_next
                    
                    non_zero_mask = S_next.abs() > mfsgd_eps_val
                    safe_reciprocal_S = torch.zeros_like(S_next)

                    if use_ones_for_nonzero_s:
                        safe_reciprocal_S[non_zero_mask] = 1.0
                    else:
                        if torch.any(non_zero_mask):
                            safe_reciprocal_S[non_zero_mask] = 1.0 / (S_next[non_zero_mask])
                        safe_reciprocal_S = torch.clamp(safe_reciprocal_S, max=max_value)
                    
                    low_rank_update_term = (U_next * safe_reciprocal_S.unsqueeze(0)) @ V_next.T

                    U_proj_comp, V_proj_comp = (U_next, V_next) if use_current_projection else (U, V)

                    # Since p.grad is zeroed by the hook, G_t_for_complement is zero.
                    # Thus, right_ortho will be zero, and eta2 term has no effect.
                    G_t_for_complement = torch.zeros_like(p.data)
                    
                    if eta2 > 0:
                        UTG_comp = U_proj_comp.t().mm(G_t_for_complement)
                        UUTG_comp = U_proj_comp.mm(UTG_comp)
                        left_ortho_comp = G_t_for_complement - UUTG_comp
                        left_ortho_V_comp = left_ortho_comp.mm(V_proj_comp)
                        left_ortho_VV_comp = left_ortho_V_comp.mm(V_proj_comp.t())
                        right_ortho = left_ortho_comp - left_ortho_VV_comp
                    else:
                        right_ortho = torch.zeros_like(p.data)

                    p_update_direction = eta1 * low_rank_update_term
                    if eta2 > 0 and torch.norm(right_ortho).item() > 0: # Should be zero
                        p_update_direction.add_(right_ortho, alpha=eta2)

                    p.data.add_(p_update_direction, alpha=-lr)

                else:
                    if p.grad is None:
                        continue
                    
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    state['step'] += 1
                    step_adam = state['step']
                    grad_adam = p.grad

                    if grad_adam is None:
                        continue

                    if adam_weight_decay != 0:
                        p.data.mul_(1 - lr * adam_weight_decay)
                    
                    exp_avg.mul_(adam_beta1).add_(grad_adam, alpha=1 - adam_beta1)
                    exp_avg_sq.mul_(adam_beta2).addcmul_(grad_adam, grad_adam.conj() if grad_adam.is_complex() else grad_adam, value=1 - adam_beta2)

                    bias_correction1 = 1 - adam_beta1 ** step_adam
                    bias_correction2 = 1 - adam_beta2 ** step_adam
                    
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(adam_eps_val)
                    update_val = exp_avg / bias_correction1

                    p.data.addcdiv_(update_val, denom, value=-lr)

    #############################################
    # Adapted rank-2r update using projections
    #############################################
    def _update_momentum_factor_from_projections(self, U, S, V, GV, GTU, UTGV, beta, target_rank):
        """
        Internal method to update (U, S, V) using pre-computed projections
        GV, GTU, UTGV, such that M_new = Proj(G_accum) + beta * (U diag(S) V^T)
        is represented by rank-`target_rank` factors (U_next, S_next, V_next).

        Args:
            U: current U factor (m, r_current)
            S: current S factor (r_current,)
            V: current V factor (n, r_current)
            GV: accumulated G @ V (m, r_current)
            GTU: accumulated G.T @ U (n, r_current)
            UTGV: accumulated U.T @ G @ V (r_current, r_current)
            beta: momentum decay factor
            target_rank: the desired rank 'r' for the output factors
        """
        m, r_current = U.shape
        n = V.shape[0]

        row_block = torch.cat([U, GV], dim=1)
        U_prime, R_U = torch.linalg.qr(row_block, mode='reduced')

        col_block = torch.cat([V, GTU], dim=1)
        V_prime, R_V = torch.linalg.qr(col_block, mode='reduced')
        
        beta_Sigma = torch.diag(beta * S)
        top_left = beta_Sigma - UTGV
        
        eye_r_current = torch.eye(r_current, device=U.device, dtype=U.dtype)
        zero_r_current = torch.zeros(r_current, r_current, device=U.device, dtype=U.dtype)

        top_row = torch.cat([top_left, eye_r_current], dim=1)
        bot_row = torch.cat([eye_r_current, zero_r_current], dim=1)
        B = torch.cat([top_row, bot_row], dim=0)

        Mid = R_U.mm(B).mm(R_V.t())

        try:
            # Using .float() for SVD stability, then casting back.
            U_dblprime_full, S_dblprime_full, V_dblprime_Vh_full = torch.linalg.svd(Mid.float())
        except Exception as e:
            print(f"SVD failed for Mid matrix. Shape: {Mid.shape}, dtype: {Mid.dtype}, device: {Mid.device}")
            print(f"Mid contains NaN: {torch.isnan(Mid).any()}, Inf: {torch.isinf(Mid).any()}")
            raise e

        r_effective_mid = S_dblprime_full.shape[0]
        r_trunc = min(target_rank, r_effective_mid)

        U_dblprime_r = U_dblprime_full[:, :r_trunc].to(dtype=U.dtype)
        S_dblprime_r = S_dblprime_full[:r_trunc].to(dtype=S.dtype)
        V_dblprime_r = V_dblprime_Vh_full.mH[:, :r_trunc].to(dtype=V.dtype)

        U_next = U_prime.mm(U_dblprime_r)
        S_next = S_dblprime_r.clone()
        V_next = V_prime.mm(V_dblprime_r)
        
        if r_trunc < target_rank:
            pad_u = torch.zeros(m, target_rank - r_trunc, device=U_next.device, dtype=U_next.dtype)
            U_next = torch.cat([U_next, pad_u], dim=1)
            pad_s = torch.zeros(target_rank - r_trunc, device=S_next.device, dtype=S_next.dtype)
            S_next = torch.cat([S_next, pad_s], dim=0)
            pad_v = torch.zeros(n, target_rank - r_trunc, device=V_next.device, dtype=V_next.dtype)
            V_next = torch.cat([V_next, pad_v], dim=1)

        return U_next, S_next, V_next


def print_mfsgd_parameter_status(model, optimizer):
    """
    Print all trainable parameters along with their MFSGD status.
    
    Args:
        model: The model with parameters
        optimizer: The MomentumFactorizedSGD optimizer
    """
    from ..train.trainer_utils import logger
    
    logger.info_rank0("\n" + "="*80)
    logger.info_rank0("TRAINABLE PARAMETERS WITH MFSGD STATUS")
    logger.info_rank0("="*80)
    
    # Get parameter groups from optimizer
    mfsgd_param_ids = set()
    for group in optimizer.param_groups:
        if group.get('is_mfsgd_group', False):
            for p in group['params']:
                mfsgd_param_ids.add(id(p))
    
    # Print all trainable parameters
    total_params = 0
    mfsgd_params = 0
    adamw_params = 0
    
    max_name_len = max([len(name) for name, _ in model.named_parameters() if _.requires_grad], default=20)
    
    logger.info_rank0(f"{'Parameter Name':<{max_name_len+5}} {'Shape':<20} {'Num Params':<15} {'MFSGD?':<10}")
    logger.info_rank0("-" * (max_name_len + 50))
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            is_mfsgd = id(param) in mfsgd_param_ids
            num_params = param.numel()
            total_params += num_params
            
            if is_mfsgd:
                mfsgd_params += num_params
            else:
                adamw_params += num_params
                
            logger.info_rank0(f"{name:<{max_name_len+5}} {str(list(param.shape)):<20} {num_params:<15,d} {'✓' if is_mfsgd else '✗'}")
    
    logger.info_rank0("-" * (max_name_len + 50))
    logger.info_rank0(f"Total parameters:      {total_params:,}")
    logger.info_rank0(f"MFSGD parameters:      {mfsgd_params:,} ({mfsgd_params/total_params*100:.2f}%)")
    logger.info_rank0(f"AdamW parameters:      {adamw_params:,} ({adamw_params/total_params*100:.2f}%)")
    logger.info_rank0("="*80 + "\n")


# Example usage (commented out, needs update for new API if used for testing)
# # ----------------------------------------------------------------------
# # Example usage:
# if __name__ == "__main__":
#     # Suppose a single 2D weight
#     w = torch.nn.Parameter(torch.randn(8, 6, requires_grad=True))
#     w_non_mfsgd = torch.nn.Parameter(torch.randn(3,3, requires_grad=True)) #
#     bias = torch.nn.Parameter(torch.randn(6, requires_grad=True))


#     # Dummy closure/ loss
#     def loss_fn():
#         # e.g., sum of w for demonstration
#         return w.sum() + w_non_mfsgd.sum() * 0.5 + bias.sum() * 0.1

#     # Instantiate the optimizer
#     # Group 0: MFSGD for 'w'
#     # Group 1: AdamW for 'w_non_mfsgd' and 'bias'
#     optim = MomentumFactorizedSGD([
#         {'params': [w], 'is_mfsgd_group': True, 'rank': 2, 'lr': 1e-2},
#         {'params': [w_non_mfsgd, bias], 'is_mfsgd_group': False, 'lr': 1e-3} # AdamW group
#     ], beta=0.9, eta1=1.0, eta2=0.1, mfsgd_eps=1e-5) # Defaults for MFSGD if not in group

#     print("Initial w:", w.data)
#     print("Initial U for w:", optim.state[w]['U'])


#     # Simple loop
#     for step_idx in range(5):
#         optim.zero_grad()
#         l = loss_fn()
#         l.backward() # This will trigger the hook for 'w'
        
#         # Check gradient and projections for 'w'
#         if 'GV' in optim.state[w]:
#             print(f"Step {step_idx} before step(): w.grad is {w.grad.norm().item() if w.grad is not None else 'None (zeroed by hook)'}, GV norm {optim.state[w]['GV'].norm().item()}")
#         else:
#             print(f"Step {step_idx} before step(): w.grad is {w.grad.norm().item() if w.grad is not None else 'None'}, GV not present")


#         optim.step()
#         print(f"Step {step_idx}: loss={l.item():.4f}")
#         print(f"  w after step: {w.data.norm().item()}")
#         if 'U' in optim.state[w]:
#             print(f"  U norm after step: {optim.state[w]['U'].norm().item()}")
#         if 'GV' not in optim.state[w]: # GV should be popped
#             print(f"  GV is correctly popped for w.")
#         else:
#             print(f"  GV still in state for w! {optim.state[w]['GV'].norm().item()}")
#         print(f"  w_non_mfsgd after step: {w_non_mfsgd.data.norm().item()}")
#         print(f"  bias after step: {bias.data.norm().item()}")

#     # Test gradient accumulation
#     print("\\nTesting gradient accumulation...")
#     optim.zero_grad()
#     loss1 = w.sum() * 0.3
#     loss1.backward(retain_graph=True) # First backward pass
#     gv_after_pass1 = optim.state[w]['GV'].clone() if 'GV' in optim.state[w] else None
#     print(f"GV after 1st backward: {gv_after_pass1.norm().item() if gv_after_pass1 is not None else 'None'}")

#     loss2 = w.sum() * 0.7
#     loss2.backward() # Second backward pass, hook accumulates
#     gv_after_pass2 = optim.state[w]['GV'].clone() if 'GV' in optim.state[w] else None
#     print(f"GV after 2nd backward: {gv_after_pass2.norm().item() if gv_after_pass2 is not None else 'None'}")

#     if gv_after_pass1 is not None and gv_after_pass2 is not None:
#         # Theory: GV from loss2 should be (GV from (grad of loss2)) + (GV from (grad of loss1))
#         # Since GV_pass1 was state['GV'] before second backward, and hook does add_
#         # gv_after_pass2 should reflect sum.
#         # This requires checking the actual values, not just norms.
#         # For a simple sum, grad(0.3*sum + 0.7*sum) = grad(sum).
#         # So GV from combined should be like GV from single grad(sum).
#         pass # Manual check of values would be needed.

#     optim.step() # Apply accumulated gradients
#     print(f"w after accumulated grad step: {w.data.norm().item()}")

#     # Test case where a MFSGD param gets no grad
#     print("\\nTesting MFSGD param with no grad...")
#     w_no_grad = torch.nn.Parameter(torch.randn(5,5,requires_grad=True))
#     optim_no_grad_test = MomentumFactorizedSGD([
#         {'params': [w_no_grad], 'is_mfsgd_group': True, 'rank': 2}
#     ])
#     # No backward pass for w_no_grad
#     optim_no_grad_test.step() # Should run without error, GV/GTU/UTGV default to zero
#     print("Step completed for param with no grad.")


#     print("\\nTest with only AdamW group")
#     p_adam = torch.nn.Parameter(torch.randn(3,3, requires_grad=True))
#     optim_adam_only = MomentumFactorizedSGD([{'params': [p_adam], 'is_mfsgd_group': False, 'lr':1e-3}])
#     optim_adam_only.zero_grad()
#     loss_adam = p_adam.sum()
#     loss_adam.backward()
#     optim_adam_only.step()
#     print("AdamW only step completed.")

#     # Test with MFSGD and eta2 > 0
#     print("\\nTest with MFSGD and eta2 > 0 (expecting right_ortho to be zero due to grad zeroing)")
#     w_eta2_test = torch.nn.Parameter(torch.randn(8, 6, requires_grad=True))
#     optim_eta2 = MomentumFactorizedSGD(
#         [{'params': [w_eta2_test], 'is_mfsgd_group': True, 'rank': 2, 'eta2': 0.5}]
#     )
#     optim_eta2.zero_grad()
#     (w_eta2_test.sum()).backward()
#     # At this point, w_eta2_test.grad is zeroed by hook.
#     # When optim_eta2.step() is called, G_t_for_complement will be zero.
#     optim_eta2.step()
#     # We expect this to run, and the update to be solely from eta1 term.
#     print(f"w_eta2_test after step with eta2 > 0: {w_eta2_test.data.norm().item()}")
#     # To verify, one could check if a non-zero eta2 changed the update compared to eta2=0.
#     # If G_t_for_complement is always zero, then eta2 should have no effect.
#     # This is a key point based on the interpretation of zeroing grad.


#     print("\\nTest done.")