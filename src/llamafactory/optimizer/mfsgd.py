import torch
from torch.optim import Optimizer
# Removed: from .mfsgd_custom_autograd import MFSGDLinear, setup_mfsgd_linear_layers

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
            try:
                # Using .float() for SVD stability, then cast back if necessary
                U_full, S_full, V_full_conj = torch.linalg.svd(p.data.float()) 
                V_full = V_full_conj.mH 
            except Exception as e:
                print(f"SVD failed during MomentumFactor init for param shape {p.shape}, dtype {p.dtype}. Error: {e}")
                r_safe = min(rank, m, n)
                self.U = torch.randn(m, r_safe, device=p.device, dtype=p.dtype)
                self.S = torch.zeros(r_safe, device=p.device, dtype=p.dtype)
                self.V = torch.randn(n, r_safe, device=p.device, dtype=p.dtype)
                if m > 0 and n > 0 and r_safe > 0:
                   self.U, _ = torch.linalg.qr(self.U)
                   self.V, _ = torch.linalg.qr(self.V)
                return

        r_trunc = min(rank, S_full.shape[0], m, n) # Ensure r_trunc respects all dimensions
        
        self.U = U_full[:, :r_trunc].clone().to(dtype=p.dtype)
        self.S = S_full[:r_trunc].clone().to(dtype=p.dtype)
        self.V = V_full[:, :r_trunc].clone().to(dtype=p.dtype)


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

    def __init__(self, params,
                 lr: float = 1e-2,
                 rank: int = 2,
                 beta: float = 0.9,
                 eta1: float = 1.0,
                 eta2: float = 1.0,
                 use_current_projection: bool = False,
                 use_ones_for_nonzero_s: bool = False,
                 mfsgd_eps: float = 1e-4,
                 nesterov: bool = False,
                 max_value: float = 10000,
                 adam_betas: tuple = (0.9, 0.999),
                 adam_eps: float = 1e-8,
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
                        # Ensure rank is not larger than parameter dimensions
                        m, n = p.shape
                        effective_rank = min(group_rank, m, n)
                        if effective_rank <=0:
                            continue # Skip if param cannot support rank > 0

                        state = self.state[p]
                        mf_init = MomentumFactor(p, effective_rank) # Use effective_rank
                        state['U'] = mf_init.U.clone().detach()
                        state['S'] = mf_init.S.clone().detach()
                        state['V'] = mf_init.V.clone().detach()
                        p.register_post_accumulate_grad_hook(self._make_post_accumulate_hook(p))

    def _make_post_accumulate_hook(self, p_param): # Renamed from _make_backward_hook
        def hook(param): # Signature changed, grad_tensor removed
            # grad is now accessed from param.grad
            grad_tensor = param.grad 
            if grad_tensor is None:
                return None

            state = self.state[p_param]
            # Ensure U and V are present
            if 'U' not in state or 'V' not in state:
                # MFSGD cannot process this parameter's gradient.
                # Set .grad to None and return None.
                param.grad = None
                return None

            U = state['U']
            V = state['V']
            
            # Ensure U, V are not empty, and grad has compatible dimensions
            if U.numel() == 0 or V.numel() == 0 or \
               (grad_tensor.shape[0] != 0 and U.shape[0] != grad_tensor.shape[0]) or \
               (grad_tensor.shape[1] != 0 and V.shape[0] != grad_tensor.shape[1]):
                # Log this potential issue if it's unexpected.
                # print(f"Skipping MFSGD projection for param due to mismatched dims or empty factors. Grad: {grad_tensor.shape}, U: {U.shape}, V: {V.shape}")
                # If we cannot process, set .grad to None and return None.
                param.grad = None
                return None

            # If grad_tensor is a zero-size tensor (e.g. shape [0, N] or [N, 0]), matmul will fail or produce zeros.
            # This can happen in some DDP settings or if a layer isn't used.
            # Projections will be zero, which is fine.
            if grad_tensor.numel() == 0:
                # Create zero tensors for gv, gtu, utgv with correct expected shapes based on U and V
                # to avoid errors in .add_ if they don't exist in state yet.
                # This assumes U and V have been initialized correctly even if grad is empty.
                # The rank `r` is U.shape[1] (or V.shape[1]).
                r = U.shape[1] if U.numel() > 0 else 0 # Get rank from U
                gv = torch.zeros(U.shape[0], r, device=U.device, dtype=U.dtype)
                gtu = torch.zeros(V.shape[0], r, device=V.device, dtype=V.dtype) # V.shape[0] is n
                utgv = torch.zeros(r, r, device=U.device, dtype=U.dtype)
            else:
                gv = torch.matmul(grad_tensor, V)
                gtu = torch.matmul(grad_tensor.t(), U)
                utgv = torch.matmul(U.t(), torch.matmul(grad_tensor, V))


            for key, proj_tensor in [('GV', gv), ('GTU', gtu), ('UTGV', utgv)]:
                if key not in state:
                    state[key] = torch.zeros_like(proj_tensor)
                state[key].add_(proj_tensor)
            
            # FOR DEBUGGING VISUALIZATION ONLY - controlled by environment variable
            import os
            if os.environ.get("MFSGD_DEBUG_EMPTY_CACHE") == "1":
                # This is very slow and should only be used for debugging memory visualization.
                torch.cuda.empty_cache()

            # Set the parameter's .grad to None to indicate it has been handled
            param.grad = None
            return None
        return hook

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group['lr']
            # rank for SVD truncation in update_momentum_factor comes from group['rank']
            # but actual factor ranks come from S.shape[0]
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
                if not p.requires_grad:
                    continue
                
                state = self.state[p]

                if apply_mfsgd_logic and p.dim() == 2 and 'U' in state: # Check 'U' to ensure MFSGD was init for p
                    U = state['U']
                    S = state['S']
                    V = state['V']

                    m, current_rank_u = U.shape 
                    n, current_rank_v = V.shape 
                    current_rank_s = S.shape[0]
                    
                    # This should be consistent due to MomentumFactor init
                    if not (current_rank_u == current_rank_s and current_rank_v == current_rank_s):
                         # Fallback to AdamW if factors are inconsistent (should not happen with proper init)
                         # print(f"Warning: Inconsistent factor ranks for param {p.shape}. U:{U.shape}, S:{S.shape}, V:{V.shape}. Using AdamW.")
                         self._apply_adamw_update(p, state, group, lr, adam_beta1, adam_beta2, adam_eps_val, adam_weight_decay)
                         continue
                    
                    current_r = current_rank_s
                    target_rank_for_svd = min(group['rank'], m, n)

                    GV = state.pop('GV', torch.zeros(m, current_r, device=U.device, dtype=U.dtype))
                    GTU = state.pop('GTU', torch.zeros(n, current_r, device=V.device, dtype=V.dtype))
                    UTGV = state.pop('UTGV', torch.zeros(current_r, current_r, device=U.device, dtype=U.dtype))

                    if nesterov:
                        pass # Simplified Nesterov

                    U_next, S_next, V_next = self._update_momentum_factor_from_projections(
                        U, S, V, GV, GTU, UTGV, beta, target_rank_for_svd
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
                        safe_reciprocal_S = torch.clamp(safe_reciprocal_S, min=-max_value, max=max_value)
                    
                    low_rank_update_term = (U_next * safe_reciprocal_S.unsqueeze(0)) @ V_next.T

                    G_t_for_complement = torch.zeros_like(p.data) # Since p.grad is set to None by the hook
                    
                    if eta2 > 0:
                        U_proj_comp, V_proj_comp = (U_next, V_next) if use_current_projection else (U, V)
                        if U_proj_comp.numel() > 0 and V_proj_comp.numel() > 0:
                            UTG_comp = U_proj_comp.t().mm(G_t_for_complement)
                            UUTG_comp = U_proj_comp.mm(UTG_comp)
                            left_ortho_comp = G_t_for_complement - UUTG_comp
                            left_ortho_V_comp = left_ortho_comp.mm(V_proj_comp)
                            left_ortho_VV_comp = left_ortho_V_comp.mm(V_proj_comp.t())
                            right_ortho = left_ortho_comp - left_ortho_VV_comp
                        else:
                             right_ortho = torch.zeros_like(p.data)
                    else:
                        right_ortho = torch.zeros_like(p.data)

                    p_update_direction = eta1 * low_rank_update_term
                    if eta2 > 0 and torch.norm(right_ortho).item() > 1e-9: 
                        p_update_direction.add_(right_ortho, alpha=eta2)
                    
                    p.data.add_(p_update_direction, alpha=-lr)

                else: # Standard AdamW logic for non-MFSGD params or if MFSGD init failed for p
                    self._apply_adamw_update(p, state, group, lr, adam_beta1, adam_beta2, adam_eps_val, adam_weight_decay)

    def _apply_adamw_update(self, p, state, group, lr, beta1, beta2, eps, weight_decay):
        # Helper for AdamW logic, extracted for clarity
        grad = p.grad
        if grad is None:
            return

        if 'step' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        state['step'] += 1
        
        if weight_decay != 0:
             p.data.mul_(1.0 - lr * weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj() if grad.is_complex() else grad, value=1 - beta2)

        step = state['step']
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
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
        (Stability improvements for QR and SVD are retained)
        """
        m, r_current_u = U.shape
        n, r_current_v = V.shape
        r_current_s = S.shape[0]

        if not (r_current_u == r_current_s and r_current_v == r_current_s):
            pass # Should be consistent from init
        
        r_current = r_current_s

        if r_current == 0:
            if S.numel() == 0:
                 beta_Sigma = torch.zeros((0,0), device=U.device, dtype=U.dtype)
            else: # S might be tensor([0.]) if rank became 0 but was 1
                 beta_Sigma = torch.diag(beta * S) 
        else:
            beta_Sigma = torch.diag(beta * S)

        row_block = torch.cat([U, GV], dim=1)
        if row_block.numel() == 0 or row_block.shape[1] == 0 :
            U_prime = torch.zeros(m, 0, device=U.device, dtype=U.dtype)
            R_U = torch.zeros(0, 0, device=U.device, dtype=U.dtype)
        else:
            original_dtype = row_block.dtype
            # QR may require float32/64 for stability, especially on CPU with float16
            try:
                U_prime_float, R_U_float = torch.linalg.qr(row_block.float(), mode='reduced')
            except Exception as e_qr: # Fallback if QR itself fails
                print(f"QR failed for row_block (U,GV): {e_qr}. Shape: {row_block.shape}, Dtype: {row_block.dtype}")
                 # Fallback to U and zeros for R_U to attempt to proceed, or handle more gracefully
                U_prime = U.clone() 
                R_U = torch.zeros(U.shape[1] + GV.shape[1], U.shape[1] + GV.shape[1], device=U.device, dtype=U.dtype) # Placeholder
                # This part may need more robust fallback (e.g. re-init U_prime from GV if U is problematic)
            else:
                U_prime = U_prime_float.to(original_dtype)
                R_U = R_U_float.to(original_dtype)

        col_block = torch.cat([V, GTU], dim=1)
        if col_block.numel() == 0 or col_block.shape[1] == 0:
            V_prime = torch.zeros(n, 0, device=V.device, dtype=V.dtype)
            R_V = torch.zeros(0, 0, device=V.device, dtype=V.dtype)
        else:
            original_dtype = col_block.dtype
            try:
                V_prime_float, R_V_float = torch.linalg.qr(col_block.float(), mode='reduced')
            except Exception as e_qr_v:
                print(f"QR failed for col_block (V,GTU): {e_qr_v}. Shape: {col_block.shape}, Dtype: {col_block.dtype}")
                V_prime = V.clone()
                R_V = torch.zeros(V.shape[1] + GTU.shape[1], V.shape[1] + GTU.shape[1], device=V.device, dtype=V.dtype)
            else:
                V_prime = V_prime_float.to(original_dtype)
                R_V = R_V_float.to(original_dtype)
        
        top_left_B = beta_Sigma - UTGV
        eye_r_current_B = torch.eye(r_current, device=U.device, dtype=U.dtype)
        zero_r_current_B = torch.zeros(r_current, r_current, device=U.device, dtype=U.dtype)

        top_row_B = torch.cat([top_left_B, eye_r_current_B], dim=1)
        bot_row_B = torch.cat([eye_r_current_B, zero_r_current_B], dim=1)
        B_matrix = torch.cat([top_row_B, bot_row_B], dim=0)

        if R_U.numel() == 0 or R_V.numel() == 0 or B_matrix.numel() == 0 or \
           R_U.shape[1] != B_matrix.shape[0] or B_matrix.shape[1] != R_V.shape[0]:
            # This indicates that one of the inputs to Mid construction is empty or dimensions are incompatible.
            # Usually happens if r_current is 0 and GV/GTU are also zero rank (e.g. from zero gradients).
            # SVD of an empty matrix or product involving empty matrix should result in empty factors.
            # Define shapes for U_dblprime etc. that will lead to 0-rank output factors correctly.
            # R_U.shape[0] or R_V.shape[0] might be the k_eff from QR.
            # If Mid cannot be formed, U_dblprime, S_dblprime, V_dblprime should be empty.
            # Let k_ru_rows be R_U.shape[0], k_rv_rows be R_V.shape[0] (these are ranks of U_prime, V_prime)
            k_ru_rows = U_prime.shape[1] # U_prime is (m, k_ru_rows)
            k_rv_rows = V_prime.shape[1] # V_prime is (n, k_rv_rows)

            U_dblprime_full = torch.zeros(k_ru_rows, 0, device=U.device, dtype=U.dtype)
            S_dblprime_full = torch.zeros(0, device=S.device, dtype=S.dtype) # S is 1D
            # V_dblprime_Vh_full is (rank, k_rv_rows). So V_dblprime_full (transposed) is (k_rv_rows, rank)
            V_dblprime_Vh_full = torch.zeros(0, k_rv_rows, device=V.device, dtype=V.dtype)
        else:
            Mid = R_U.mm(B_matrix).mm(R_V.t())
            try:
                U_dblprime_full_float, S_dblprime_full_float, V_dblprime_Vh_full_float = torch.linalg.svd(Mid.float())
                U_dblprime_full = U_dblprime_full_float.to(U.dtype)
                S_dblprime_full = S_dblprime_full_float.to(S.dtype)
                V_dblprime_Vh_full = V_dblprime_Vh_full_float.to(V.dtype)
            except Exception as e_svd:
                print(f"SVD failed for Mid matrix. Shape: {Mid.shape}, dtype: {Mid.dtype}, device: {Mid.device}. Error: {e_svd}")
                print(f"Mid contains NaN: {torch.isnan(Mid).any()}, Inf: {torch.isinf(Mid).any()}")
                if target_rank == 0: # If target is rank 0, return empty factors.
                    return (torch.zeros(m, 0, device=U.device, dtype=U.dtype),
                            torch.zeros(0, device=S.device, dtype=S.dtype),
                            torch.zeros(n, 0, device=V.device, dtype=V.dtype))
                # Fallback: attempt to return existing U, S*beta, V to simulate decay
                # This is a temporary measure if SVD fails, to avoid crashing.
                # print(f"Critical SVD Error. Attempting to return decayed U,S,V. THIS IS A FALLBACK.")
                # return U.clone(), S.clone() * beta, V.clone() # Risky if S has zeros or target_rank mismatch
                raise e_svd # Re-raise to make the SVD failure visible

        r_effective_mid = S_dblprime_full.shape[0]
        r_trunc = min(target_rank, r_effective_mid)

        U_dblprime_r = U_dblprime_full[:, :r_trunc]
        S_dblprime_r = S_dblprime_full[:r_trunc].clone()
        # V_dblprime_Vh is (r_trunc, k_rv_rows), so V_dblprime_r is (k_rv_rows, r_trunc)
        V_dblprime_r = V_dblprime_Vh_full[:r_trunc, :].mH 

        if U_prime.shape[1] == 0 and U_dblprime_r.shape[0] == 0 :
            U_next = torch.zeros(m, r_trunc, device=U.device, dtype=U.dtype)
        elif U_prime.shape[1] != U_dblprime_r.shape[0]:
            U_next = torch.zeros(m, r_trunc, device=U.device, dtype=U.dtype)
        else:
            U_next = U_prime.mm(U_dblprime_r)

        if V_prime.shape[1] == 0 and V_dblprime_r.shape[0] == 0 :
            V_next = torch.zeros(n, r_trunc, device=V.device, dtype=V.dtype)
        elif V_prime.shape[1] != V_dblprime_r.shape[0]:
            V_next = torch.zeros(n, r_trunc, device=V.device, dtype=V.dtype)
        else:
            V_next = V_prime.mm(V_dblprime_r)
            
        S_next = S_dblprime_r

        if r_trunc < target_rank:
            pad_width = target_rank - r_trunc
            pad_u_tensor = torch.zeros(m, pad_width, device=U_next.device, dtype=U_next.dtype)
            U_next = torch.cat([U_next, pad_u_tensor], dim=1)
            pad_s_tensor = torch.zeros(pad_width, device=S_next.device, dtype=S_next.dtype)
            S_next = torch.cat([S_next, pad_s_tensor], dim=0)
            pad_v_tensor = torch.zeros(n, pad_width, device=V_next.device, dtype=V_next.dtype)
            V_next = torch.cat([V_next, pad_v_tensor], dim=1)

        return U_next, S_next, V_next


def print_mfsgd_parameter_status(model, optimizer):
    """
    Print all trainable parameters along with their MFSGD status.
    
    Args:
        model: The model with parameters
        optimizer: The MomentumFactorizedSGD optimizer
    """
    try:
        from ..train.trainer_utils import logger 
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
        # Basic fallback if info_rank0 not present, e.g. if logger doesn't have it
        if not hasattr(logger, 'info_rank0'):
            logger.info_rank0 = logger.info
    
    logger.info_rank0("\n" + "="*80)
    logger.info_rank0("TRAINABLE PARAMETERS WITH MFSGD STATUS")
    logger.info_rank0("="*80)
    
    mfsgd_param_ids = set()
    for group in optimizer.param_groups:
        if group.get('is_mfsgd_group', False):
            for p in group['params']:
                # Check if MFSGD was actually initialized for this param (e.g. U,S,V exist in state)
                if p.requires_grad and p in optimizer.state and 'U' in optimizer.state[p]:
                    mfsgd_param_ids.add(id(p))
    
    total_params_val = 0 # Renamed to avoid conflict
    mfsgd_params_val = 0
    adamw_params_val = 0
    
    param_infos = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            is_mfsgd = id(param) in mfsgd_param_ids
            num_p = param.numel()
            total_params_val += num_p
            
            if is_mfsgd:
                mfsgd_params_val += num_p
            else:
                adamw_params_val += num_p
            param_infos.append({"name": name, "shape": list(param.shape), "num_params": num_p, "is_mfsgd": is_mfsgd})

    max_name_len = max([len(info["name"]) for info in param_infos] + [20]) if param_infos else 20
    
    logger.info_rank0(f"{'Parameter Name':<{max_name_len+5}} {'Shape':<20} {'Num Params':<15} {'MFSGD?':<10}")
    logger.info_rank0("-" * (max_name_len + 50))
    
    for info in param_infos:
        logger.info_rank0(f"{info['name']:<{max_name_len+5}} {str(info['shape']):<20} {info['num_params']:<15,d} {'✓' if info['is_mfsgd'] else '✗'}")
    
    logger.info_rank0("-" * (max_name_len + 50))
    logger.info_rank0(f"Total trainable parameters: {total_params_val:,}")
    if total_params_val > 0:
        logger.info_rank0(f"MFSGD parameters:         {mfsgd_params_val:,} ({mfsgd_params_val/total_params_val*100:.2f}%)")
        logger.info_rank0(f"AdamW parameters:         {adamw_params_val:,} ({adamw_params_val/total_params_val*100:.2f}%)")
    else:
        logger.info_rank0("MFSGD parameters:         0 (N/A)")
        logger.info_rank0("AdamW parameters:         0 (N/A)")
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
#     if w in optim.state and 'U' in optim.state[w]:
#        print("Initial U for w:", optim.state[w]['U'])


#     # Simple loop
#     for step_idx in range(5):
#         optim.zero_grad()
#         l = loss_fn()
#         l.backward() # This will trigger the hook for 'w'
        
#         # Check gradient and projections for 'w'
#         if w in optim.state and 'GV' in optim.state[w]:
#             # w.grad should be None or zero due to hook
#             grad_norm_msg = "None (zeroed by hook)" if w.grad is None or w.grad.norm().item() < 1e-6 else w.grad.norm().item()
#             print(f"Step {step_idx} before step(): w.grad is {grad_norm_msg}, GV norm {optim.state[w]['GV'].norm().item()}")
#         elif w.grad is not None:
#             print(f"Step {step_idx} before step(): w.grad is {w.grad.norm().item()}, GV not present (param not MFSGD or no grad yet)")
#         else:
#             print(f"Step {step_idx} before step(): w.grad is None, GV not present")


#         optim.step()
#         print(f"Step {step_idx}: loss={l.item():.4f}")
#         print(f"  w after step: {w.data.norm().item()}")
#         if w in optim.state and 'U' in optim.state[w]:
#             print(f"  U norm after step: {optim.state[w]['U'].norm().item()}")
#         if w in optim.state and 'GV' not in optim.state[w]: # GV should be popped
#             print(f"  GV is correctly popped for w.")
#         elif w in optim.state and 'GV' in optim.state[w]: # Should not happen if MFSGD step occurred
#             print(f"  GV still in state for w! {optim.state[w]['GV'].norm().item()}")
#         print(f"  w_non_mfsgd after step: {w_non_mfsgd.data.norm().item()}")
#         print(f"  bias after step: {bias.data.norm().item()}")

#     # Test gradient accumulation
#     print("\nTesting gradient accumulation...")
#     optim.zero_grad()
#     loss1 = w.sum() * 0.3
#     loss1.backward(retain_graph=True) # First backward pass
#     gv_after_pass1 = optim.state[w]['GV'].clone() if w in optim.state and 'GV' in optim.state[w] else None
#     print(f"GV after 1st backward: {gv_after_pass1.norm().item() if gv_after_pass1 is not None else 'None'}")

#     loss2 = w.sum() * 0.7
#     loss2.backward() # Second backward pass, hook accumulates
#     gv_after_pass2 = optim.state[w]['GV'].clone() if w in optim.state and 'GV' in optim.state[w] else None
#     print(f"GV after 2nd backward: {gv_after_pass2.norm().item() if gv_after_pass2 is not None else 'None'}")

#     # With retain_graph=True and multiple backwards, GV state should sum up.
#     # A single grad from (loss1+loss2) would be grad(w.sum()). Hook should reflect this sum.
#     # Example: if grad(w.sum()) leads to GV_total,
#     # GV from 0.3*w.sum() is 0.3*GV_total. GV from 0.7*w.sum() is 0.7*GV_total.
#     # Accumulated state['GV'] should be (0.3*GV_total + 0.7*GV_total) = GV_total.
#     # This can be manually verified if needed.

#     optim.step() # Apply accumulated gradients
#     print(f"w after accumulated grad step: {w.data.norm().item()}")

#     # Test case where a MFSGD param gets no grad
#     print("\nTesting MFSGD param with no grad...")
#     w_no_grad = torch.nn.Parameter(torch.randn(5,5,requires_grad=True))
#     optim_no_grad_test = MomentumFactorizedSGD([
#         {'params': [w_no_grad], 'is_mfsgd_group': True, 'rank': 2}
#     ])
#     # No backward pass for w_no_grad, so hook is not called, GV/GTU/UTGV not created in state.
#     optim_no_grad_test.step() # Should run without error, GV/GTU/UTGV default to zero via state.pop()
#     print("Step completed for param with no grad.")


#     print("\nTest with only AdamW group")
#     p_adam = torch.nn.Parameter(torch.randn(3,3, requires_grad=True))
#     optim_adam_only = MomentumFactorizedSGD([{'params': [p_adam], 'is_mfsgd_group': False, 'lr':1e-3}])
#     optim_adam_only.zero_grad()
#     loss_adam = p_adam.sum()
#     loss_adam.backward()
#     optim_adam_only.step()
#     print("AdamW only step completed.")

#     # Test with MFSGD and eta2 > 0
#     print("\nTest with MFSGD and eta2 > 0 (expecting right_ortho to be zero due to grad zeroing)")
#     w_eta2_test = torch.nn.Parameter(torch.randn(8, 6, requires_grad=True))
#     optim_eta2 = MomentumFactorizedSGD(
#         [{'params': [w_eta2_test], 'is_mfsgd_group': True, 'rank': 2, 'eta2': 0.5}]
#     )
#     optim_eta2.zero_grad()
#     (w_eta2_test.sum()).backward()
#     # At this point, w_eta2_test.grad is zeroed by hook.
#     # When optim_eta2.step() is called, G_t_for_complement will be zero.
#     optim_eta2.step()
#     print(f"w_eta2_test after step with eta2 > 0: {w_eta2_test.data.norm().item()}")

#     print("\nTest done.")
# # End of example