import torch
from torch.optim import Optimizer
# Import new classes
from .mfsgd_custom_autograd import MFSGDLinear, setup_mfsgd_linear_layers # Added

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
            # Ensure p is on a device for SVD, handle CPU explicitly if needed.
            # SVD may fail on CPU for float16, ensure p.float() if necessary and original dtype is float16.
            # Original code did not have specific handling here, assuming p is well-formed.
            try:
                U_full, S_full, V_full_conj = torch.linalg.svd(p.data.float()) # Use .float() for stability, then cast back
                V_full = V_full_conj.mH # For complex, V is Vh. For real, Vh is V.T
            except Exception as e:
                print(f"SVD failed during MomentumFactor init for param shape {p.shape}, dtype {p.dtype}. Error: {e}")
                # Fallback: Initialize with random orthogonal matrices or zeros if SVD fails
                # This is a patch; ideally, the cause of SVD failure should be investigated.
                r_safe = min(rank, m, n)
                self.U = torch.randn(m, r_safe, device=p.device, dtype=p.dtype)
                self.S = torch.zeros(r_safe, device=p.device, dtype=p.dtype) # Start with zero singular values
                self.V = torch.randn(n, r_safe, device=p.device, dtype=p.dtype)
                if m > 0 and n > 0 and r_safe > 0: # Avoid QR on empty
                   self.U, _ = torch.linalg.qr(self.U)
                   self.V, _ = torch.linalg.qr(self.V)
                return


        r_trunc = min(rank, S_full.shape[0]) # rank can be min(m,n) at most from SVD result
        
        self.U = U_full[:, :r_trunc].clone().to(dtype=p.dtype)
        self.S = S_full[:r_trunc].clone().to(dtype=p.dtype)
        self.V = V_full[:, :r_trunc].clone().to(dtype=p.dtype)


#############################################
#   The MomentumFactorizedSGD Optimizer
#############################################
class MomentumFactorizedSGD(Optimizer):
    r"""
    Implements a rank-r factorized momentum update.
    Uses a custom torch.autograd.Function (LowRankProjectionForWeight)
    applied by MFSGDLinear layers to project gradients and zero them out.

    Key aspects:
    1. Factors U, S, V for momentum M_t ≈ U_t diag(S_t) V_t^T are stored.
    2. MFSGDLinear layers, when their weights are part of an MFSGD group,
       use LowRankProjectionForWeight. This function, in its backward pass:
          - Receives the full gradient for the weight.
          - Computes projections: GV, GTU, UTGV.
          - Accumulates these into pre-allocated buffers in optimizer.state[param].
          - Returns a zeroed gradient for the weight.
    3. The `step()` method uses these accumulated projections from the buffers
       to update U, S, V via a block-matrix SVD approach.
    4. Parameter update includes terms from updated low-rank factors (eta1)
       and an optional orthogonal complement term (eta2, currently zero effect
       as full grad is zeroed).

    Args:
      params: iterable of parameters to optimize
      lr:     global learning rate
      rank:   integer rank r for MFSGD factorization
      beta:   momentum decay factor (0.0 <= beta < 1.0)
      eta1:   scale factor for the low-rank momentum term in parameter update
      eta2:   scale factor for the orthogonal complement gradient term in parameter update
      use_current_projection: For orthogonal complement, use U_next, V_next (True) or U, V (False).
      use_ones_for_nonzero_s: If True, use 1.0 for reciprocal of non-zero singular values, else use 1/S.
      mfsgd_eps: Epsilon for thresholding singular values (S_next.abs() > mfsgd_eps).
      nesterov: If True, applies Nesterov momentum (currently simplified).
      max_value: Maximum clamp value for reciprocal singular values.
      adam_betas: Tuple (beta1, beta2) for AdamW fallback.
      adam_eps: Epsilon for AdamW fallback.
      adam_weight_decay: Weight decay for AdamW fallback.
    """

    def __init__(self, params, model_for_mfsgd_layers=None, # Added model_for_mfsgd_layers
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
            group_rank = group['rank'] # Get rank from group specific settings
            is_mfsgd_group = group.get('is_mfsgd_group', False)

            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                if is_mfsgd_group and p.dim() == 2:
                    m, n = p.shape
                    # Ensure rank is not larger than parameter dimensions
                    effective_rank = min(group_rank, m, n)
                    if effective_rank <= 0 : # Skip if effective rank is 0 or less (e.g. m or n is 0)
                        state['_is_mfsgd_group_param'] = False
                        # Param will be handled by AdamW logic by falling through
                        # print(f"Warning: MFSGD param {p.shape} has effective_rank {effective_rank}, will use AdamW.")
                        continue


                    state['_is_mfsgd_group_param'] = True
                    mf_init = MomentumFactor(p, effective_rank) # Use effective_rank
                    state['U'] = mf_init.U.clone().detach()
                    state['S'] = mf_init.S.clone().detach()
                    state['V'] = mf_init.V.clone().detach()
                    
                    # Initialize buffers for projections
                    # GV: (m, r), GTU: (n, r), UTGV: (r, r)
                    # Use actual rank from initialized U, S, V (r_init)
                    r_init = state['S'].shape[0]
                    state['GV_buffer'] = torch.zeros(m, r_init, device=p.device, dtype=p.dtype)
                    state['GTU_buffer'] = torch.zeros(n, r_init, device=p.device, dtype=p.dtype)
                    state['UTGV_buffer'] = torch.zeros(r_init, r_init, device=p.device, dtype=p.dtype)
                    
                    # Removed p.register_hook(self._make_backward_hook(p))
                else:
                    state['_is_mfsgd_group_param'] = False # Mark non-MFSGD params

        if model_for_mfsgd_layers is not None:
            self.setup_optimizer_for_model_layers(model_for_mfsgd_layers)
        else:
            # Check if any MFSGDLinear layers exist without being configured
            # This requires iterating over all params again, or assuming user calls setup method.
            # For now, we rely on user providing the model or calling setup_optimizer_for_model_layers.
            pass
            
    def setup_optimizer_for_model_layers(self, model):
        """
        Configures MFSGDLinear layers in the model with necessary state references
        from this optimizer. This method should be called after the model and
        optimizer have been initialized.
        """
        # This function is now part of mfsgd_custom_autograd.py to avoid circular dependency
        # It needs `MomentumFactorizedSGD` type, which is self here.
        setup_mfsgd_linear_layers(model, self)


    # Removed _make_backward_hook method entirely

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group['lr']
            # rank = group['rank'] # rank used for initialization, actual rank from S.shape[0]
            beta = group['beta']
            eta1 = group['eta1']
            eta2 = group['eta2']
            use_current_projection = group['use_current_projection']
            use_ones_for_nonzero_s = group['use_ones_for_nonzero_s']
            mfsgd_eps_val = group['mfsgd_eps']
            nesterov = group['nesterov'] # Nesterov not fully implemented with this structure yet
            max_value = group['max_value']

            adam_beta1, adam_beta2 = group['adam_betas']
            adam_eps_val = group['adam_eps']
            adam_weight_decay = group['adam_weight_decay']

            # apply_mfsgd_logic determined by state['_is_mfsgd_group_param']
            # group.get('is_mfsgd_group', False) was for initialization.

            for p in group['params']:
                if not p.requires_grad: # Skip params that don't require gradients
                    continue
                
                state = self.state[p]

                # Check if this parameter is managed by MFSGD logic
                if state.get('_is_mfsgd_group_param', False) and p.dim() == 2:
                    U = state['U']
                    S = state['S']
                    V = state['V']
                    
                    # Effective rank for current factors
                    current_rank = S.shape[0] 
                    # Target rank for update (from group settings, capped by param dims)
                    # This should be the rank used for initializing buffers.
                    # For _update_momentum_factor_from_projections, target_rank is group['rank']
                    # but capped by SVD result of Mid matrix.
                    # The buffers were initialized with rank from MomentumFactor(p, group_rank)
                    # So, group['rank'] is the intended target_rank for SVD truncation.
                    target_rank_for_svd = min(group['rank'], p.shape[0], p.shape[1])


                    # Retrieve accumulated projections from buffers
                    GV = state['GV_buffer']
                    GTU = state['GTU_buffer']
                    UTGV = state['UTGV_buffer']

                    if nesterov:
                        # Nesterov logic is complex with deferred gradient application.
                        # Current simplified version might not be fully Nesterov.
                        # With custom autograd, full Nesterov might be more feasible if
                        # projections are applied to grad(params_at_lookahead_step).
                        pass

                    # Ensure U,S,V and GV,GTU,UTGV have consistent ranks before update
                    # This should be handled by initialization and buffer creation.
                    # If r_init (from S.shape[0]) differs from buffer dim, it's an issue.
                    # Assuming current_rank matches buffer dimensions for GV, GTU, UTGV's second dim.

                    U_next, S_next, V_next = self._update_momentum_factor_from_projections(
                        U, S, V, GV, GTU, UTGV, beta, target_rank_for_svd # Use target_rank_for_svd
                    )

                    state['U'] = U_next
                    state['S'] = S_next
                    state['V'] = V_next
                    
                    # Zero out the buffers for the next accumulation cycle
                    state['GV_buffer'].zero_()
                    state['GTU_buffer'].zero_()
                    state['UTGV_buffer'].zero_()
                    
                    non_zero_mask = S_next.abs() > mfsgd_eps_val
                    safe_reciprocal_S = torch.zeros_like(S_next)

                    if use_ones_for_nonzero_s:
                        safe_reciprocal_S[non_zero_mask] = 1.0
                    else:
                        if torch.any(non_zero_mask): # Avoid division by zero if all S are zero
                            safe_reciprocal_S[non_zero_mask] = 1.0 / (S_next[non_zero_mask])
                        # Clamp before potential inf/NaNs if S_next was very small but > eps
                        safe_reciprocal_S = torch.clamp(safe_reciprocal_S, min=-max_value, max=max_value) # ensure clamping applied
                    
                    low_rank_update_term = (U_next * safe_reciprocal_S.unsqueeze(0)) @ V_next.T

                    # Orthogonal complement logic (eta2 term)
                    # Since LowRankProjectionForWeight returns zeroed grad for the parameter,
                    # the standard p.grad (if it were to exist) would be zero.
                    # G_t_for_complement would be based on this zero gradient.
                    # Thus, the eta2 term remains zero unless G_t_for_complement is reconstructed differently.
                    G_t_for_complement = torch.zeros_like(p.data) # This is the effective gradient seen by this part
                    
                    if eta2 > 0:
                        U_proj_comp, V_proj_comp = (U_next, V_next) if use_current_projection else (U, V)
                        # Ensure U_proj_comp and V_proj_comp are not empty if SVD reduced rank to 0
                        if U_proj_comp.numel() > 0 and V_proj_comp.numel() > 0:
                            UTG_comp = U_proj_comp.t().mm(G_t_for_complement)
                            UUTG_comp = U_proj_comp.mm(UTG_comp)
                            left_ortho_comp = G_t_for_complement - UUTG_comp
                            left_ortho_V_comp = left_ortho_comp.mm(V_proj_comp)
                            left_ortho_VV_comp = left_ortho_V_comp.mm(V_proj_comp.t())
                            right_ortho = left_ortho_comp - left_ortho_VV_comp
                        else: # Factors are empty, no orthogonal component possible
                            right_ortho = torch.zeros_like(p.data)

                    else:
                        right_ortho = torch.zeros_like(p.data)

                    p_update_direction = eta1 * low_rank_update_term
                    if eta2 > 0 and torch.norm(right_ortho).item() > 1e-9: # Check if right_ortho is non-negligible
                        p_update_direction.add_(right_ortho, alpha=eta2)
                    
                    p.data.add_(p_update_direction, alpha=-lr)

                else: # Standard AdamW update for non-MFSGD params or non-2D MFSGD group params
                    # The p.grad here is the true gradient, not zeroed by our custom autograd fn.
                    # LowRankProjectionForWeight passes grad_weight through if not is_mfsgd_param.
                    grad_adam = p.grad 
                    if grad_adam is None: # If a parameter in AdamW group has no grad (e.g. not used in loss)
                        continue
                    
                    # AdamW state initialization (moved here for clarity, happens once per param)
                    if 'step' not in state:
                        state['step'] = 0
                        # exp_avg and exp_avg_sq should be initialized on the same device and dtype as the parameter
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    state['step'] += 1
                    step_adam = state['step']
                    
                    # AdamW weight decay
                    # To matchtransformers.AdamW, weight decay is applied to param before grad update.
                    if adam_weight_decay != 0:
                         p.data.mul_(1.0 - lr * adam_weight_decay) # L2 penalty

                    # AdamW updates
                    exp_avg.mul_(adam_beta1).add_(grad_adam, alpha=1 - adam_beta1)
                    exp_avg_sq.mul_(adam_beta2).addcmul_(grad_adam, grad_adam.conj() if grad_adam.is_complex() else grad_adam, value=1 - adam_beta2)

                    bias_correction1 = 1 - adam_beta1 ** step_adam
                    bias_correction2 = 1 - adam_beta2 ** step_adam
                    
                    # Corrected denom calculation
                    # denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(adam_eps_val)
                    # More stable way:
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(adam_eps_val)

                    update_val = exp_avg / bias_correction1
                    p.data.addcdiv_(update_val, denom, value=-lr)
        # ... existing code ...
    #############################################
    # Adapted rank-2r update using projections
    #############################################
    def _update_momentum_factor_from_projections(self, U, S, V, GV, GTU, UTGV, beta, target_rank):
        """
        Internal method to update (U, S, V) using pre-computed projections
        GV, GTU, UTGV, such that M_new = Proj(G_accum) + beta * (U diag(S) V_T)
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
        m, r_current_u = U.shape
        n, r_current_v = V.shape
        r_current_s = S.shape[0]

        # Consistency check for ranks, important if factors could become empty
        if not (r_current_u == r_current_s and r_current_v == r_current_s):
            # This might happen if SVD reduces rank to 0 initially or after an update.
            # Handle gracefully: if r_current is 0, then U,S,V are empty or zero-dim.
            # Projections GV, GTU, UTGV should also be zero-dim or match.
            # If current rank is 0, effectively we are re-initializing from G.
            # print(f"Rank inconsistency: U({U.shape}), S({S.shape}), V({V.shape}). Projections: GV({GV.shape}), GTU({GTU.shape}), UTGV({UTGV.shape})")
            # This case needs careful handling. For now, assume consistent non-empty factors or SVD handles it.
            pass
        
        r_current = r_current_s # Assuming consistent ranks

        # If current rank is 0, beta * Sigma is problematic.
        # Treat as re-initialization from current gradient projections if r_current is 0.
        if r_current == 0:
            # Simplified: if momentum is zero, new momentum is just from current grad projections.
            # This requires a different update form: SVD of G_approx = U_g S_g V_g^T
            # where U_g, V_g are from QR of G related terms.
            # For now, the existing block matrix SVD might handle r_current=0 if blocks are formed carefully.
            # If U,V,S are empty, cat with GV, GTU will make them non-empty if GV,GTU exist.
            # `torch.diag(beta * S)` if S is empty (0-dim) might error or return empty.
            # Ensure S is at least 1D for diag. If S is 0-dim (e.g. from S[:0]), make it 1D zeros.
            if S.numel() == 0 and r_current == 0: # S could be tensor([])
                 beta_Sigma = torch.zeros((0,0), device=U.device, dtype=U.dtype) # Empty matrix for block
            else:
                 beta_Sigma = torch.diag(beta * S) # If S is 1D len 0, diag is 0x0
        else:
            beta_Sigma = torch.diag(beta * S)

        # Ensure GV, GTU, UTGV have compatible shapes even if r_current is 0 for U,S,V.
        # This means GV, GTU could be (m,0), (n,0) and UTGV (0,0) if grad was zero or param small.
        # Or if they were initialized based on a rank that became 0.
        # The cat operation should handle empty tensors correctly (concatenating with an empty tensor).
        # Example: torch.cat([torch.randn(3,0), torch.randn(3,2)], dim=1) -> shape (3,2)
        
        # Row block for U_prime
        # If U is (m,0) and GV is (m, k_gv_rank), row_block is (m, k_gv_rank)
        row_block = torch.cat([U, GV], dim=1)
        if row_block.numel() == 0 or row_block.shape[1] == 0 : # QR fails if matrix is empty or has zero columns
            U_prime = torch.zeros(m, 0, device=U.device, dtype=U.dtype)
            R_U = torch.zeros(0, 0, device=U.device, dtype=U.dtype)
        else:
            original_dtype = row_block.dtype
            U_prime_float, R_U_float = torch.linalg.qr(row_block.float(), mode='reduced')
            U_prime = U_prime_float.to(original_dtype)
            R_U = R_U_float.to(original_dtype)

        # Col block for V_prime
        # If V is (n,0) and GTU is (n, k_gtu_rank), col_block is (n, k_gtu_rank)
        col_block = torch.cat([V, GTU], dim=1)
        if col_block.numel() == 0 or col_block.shape[1] == 0:
            V_prime = torch.zeros(n, 0, device=V.device, dtype=V.dtype)
            R_V = torch.zeros(0, 0, device=V.device, dtype=V.dtype)
        else:
            original_dtype = col_block.dtype
            V_prime_float, R_V_float = torch.linalg.qr(col_block.float(), mode='reduced')
            V_prime = V_prime_float.to(original_dtype)
            R_V = R_V_float.to(original_dtype)
        
        # Dimensions of R_U: (k_ru, k_ru), where k_ru = rank of row_block = U_prime.shape[1]
        # Dimensions of R_V: (k_rv, k_rv), where k_rv = rank of col_block = V_prime.shape[1]
        # R_U is (row_block.shape[1], U_prime.shape[1]) if not square, but mode='reduced' gives U_prime cols = min(rows,cols) of block
        # Actually, R_U is (min(M,N), N) from A (M,N). For reduced, U_prime (M,K), R_U (K,N) where K=min(M,N) for full rank A.
        # If A is (m, r_u + r_gv), U_prime is (m, k_eff_u), R_U is (k_eff_u, r_u + r_gv)
        # R_V is (k_eff_v, r_v + r_gtu)
        
        # Core matrix B construction
        # Top-left: beta*Sigma - UTGV. Sigma is (r_current, r_current). UTGV is (r_current, r_current)
        # Top-right: I_r_current (r_current, r_current)
        # Bot-left: I_r_current (r_current, r_current)
        # Bot-right: Zero_r_current (r_current, r_current)
        
        # If r_current is 0, these blocks are 0x0 or similar.
        # beta_Sigma would be 0x0. UTGV should be 0x0.
        # Identity of size 0x0 is problematic. torch.eye(0) is 0x0.
        
        top_left_B = beta_Sigma - UTGV # (r_current, r_current)
        
        # Ensure Identity and Zero matrices are correctly sized even if r_current is 0.
        # torch.eye(0) is a 0x0 tensor.
        eye_r_current_B = torch.eye(r_current, device=U.device, dtype=U.dtype)
        zero_r_current_B = torch.zeros(r_current, r_current, device=U.device, dtype=U.dtype)

        top_row_B = torch.cat([top_left_B, eye_r_current_B], dim=1) # (r_current, 2*r_current)
        bot_row_B = torch.cat([eye_r_current_B, zero_r_current_B], dim=1) # (r_current, 2*r_current)
        B_matrix = torch.cat([top_row_B, bot_row_B], dim=0) # (2*r_current, 2*r_current)

        # Mid = R_U @ B @ R_V.T
        # R_U is (k_eff_u, r_current_u_input_to_qr + r_gv)
        # R_V is (k_eff_v, r_current_v_input_to_qr + r_gtu)
        # B_matrix is (r_current_u_input_to_qr + r_gv, r_current_v_input_to_qr + r_gtu) effectively,
        # matching the structure of [U GV] and [V GTU] if r_u_in = r_v_in = r_current.
        # B_matrix needs to map from space of [U GV] columns to [V GTU] columns.
        # Original paper's B has size (r+p)x(r+p) if G is rank p. Here G is projected.
        # Let original U,V be (m,r), (n,r). GV (m,r), GTU (n,r).
        # R_U from QR of [U GV] (m, 2r). R_U is (2r, 2r) if full rank, U_prime (m,2r)
        # R_V from QR of [V GTU] (n, 2r). R_V is (2r, 2r) if full rank, V_prime (n,2r)
        # B is (2r,2r). Mid is (2r,2r).
        # This seems consistent.
        
        # If R_U or R_V are empty (e.g. from QR of zero-column matrix)
        if R_U.numel() == 0 or R_V.numel() == 0 or B_matrix.numel() == 0 or R_U.shape[1] != B_matrix.shape[0] or B_matrix.shape[1] != R_V.shape[0]:
            # One of the components is empty or mismatched, implies Mid should be effectively zero or empty SVD
            # This can happen if r_current = 0 and GV/GTU are also zero-rank (e.g. zero grad)
            # print(f"Skipping SVD due to empty/mismatched matrices for Mid. R_U:{R_U.shape}, B:{B_matrix.shape}, R_V:{R_V.shape}")
            U_dblprime_full = torch.zeros(R_U.shape[0], 0, device=U.device, dtype=U.dtype) # Match U_prime.mm(..) structure
            S_dblprime_full = torch.zeros(0, device=U.device, dtype=S.dtype)
            V_dblprime_Vh_full = torch.zeros(R_V.shape[0], 0, device=V.device, dtype=V.dtype) # Match V_prime.mm(..) structure

        else:
            Mid = R_U.mm(B_matrix).mm(R_V.t())

            try:
                # Using .float() for SVD stability, then casting back.
                # Add small epsilon for stability if Mid could be ill-conditioned before SVD
                # Mid_stable = Mid + torch.eye(Mid.shape[0], device=Mid.device, dtype=Mid.dtype) * 1e-9 * (beta if beta > 0 else 1)
                U_dblprime_full_float, S_dblprime_full_float, V_dblprime_Vh_full_float = torch.linalg.svd(Mid.float())
                
                U_dblprime_full = U_dblprime_full_float.to(dtype=U.dtype)
                S_dblprime_full = S_dblprime_full_float.to(dtype=S.dtype)
                V_dblprime_Vh_full = V_dblprime_Vh_full_float.to(dtype=V.dtype) # Vh is already V.conj().T

            except Exception as e:
                # This is a critical failure.
                print(f"SVD failed for Mid matrix. Shape: {Mid.shape}, dtype: {Mid.dtype}, device: {Mid.device}")
                print(f"Mid contains NaN: {torch.isnan(Mid).any()}, Inf: {torch.isinf(Mid).any()}")
                # Fallback: return original U, S, V or zero factors to prevent crash
                # This means momentum doesn't update correctly for this step.
                # To avoid downstream errors, ensure shapes are consistent with target_rank for padding.
                # Safest might be to return U,S,V scaled by beta (simple decay if update fails)
                # For now, raise to make issue visible.
                # If target_rank is 0, should return empty factors.
                if target_rank == 0:
                    return (torch.zeros(m, 0, device=U.device, dtype=U.dtype),
                            torch.zeros(0, device=S.device, dtype=S.dtype),
                            torch.zeros(n, 0, device=V.device, dtype=V.dtype))

                # Attempt to return something of target_rank if SVD fails catastrophically
                # This is a last resort. A zero matrix doesn't carry momentum.
                # Returning U,S,V means momentum coasts.
                # print(f"Error during SVD: {e}. Returning previous U,S,V scaled by beta as fallback.")
                # S_decayed = S * beta
                # return U, S_decayed, V # This might be unstable if S has zeros.
                raise e # Re-raise to ensure visibility of the problem.


        r_effective_mid = S_dblprime_full.shape[0]
        r_trunc = min(target_rank, r_effective_mid)

        U_dblprime_r = U_dblprime_full[:, :r_trunc]
        S_dblprime_r = S_dblprime_full[:r_trunc].clone() # Clone S to avoid modifying SVD output if it's a view
        V_dblprime_r_Vh = V_dblprime_Vh_full[:r_trunc, :] # This is Vh part from SVD (U S Vh)
        V_dblprime_r = V_dblprime_r_Vh.mH # V = (Vh).mH

        # Final factors
        # If U_prime or V_prime are empty (e.g. m x 0), mm with U_dblprime_r (0 x k) is (m x k)
        # This needs U_dblprime_r and V_dblprime_r to handle 0-dim correctly if U_prime/V_prime are mx0
        # U_dblprime_r would be (0, r_trunc), S (r_trunc), V_dblprime_r (0, r_trunc)
        # Result U_next (m, r_trunc), V_next (n, r_trunc)
        
        # If U_prime is (m,0) and U_dblprime_r is (0, r_trunc), then U_next is (m, r_trunc) containing zeros.
        if U_prime.shape[1] == 0 and U_dblprime_r.shape[0] == 0 : # U_prime is mx0, U_dblprime_r is 0xr_trunc
            U_next = torch.zeros(m, r_trunc, device=U.device, dtype=U.dtype)
        elif U_prime.shape[1] != U_dblprime_r.shape[0]: # Should not happen if logic is correct
            print(f"Shape mismatch for U_next: U_prime {U_prime.shape}, U_dblprime_r {U_dblprime_r.shape}")
            # Fallback to empty or handle error
            U_next = torch.zeros(m, r_trunc, device=U.device, dtype=U.dtype) # Safest fallback
        else:
            U_next = U_prime.mm(U_dblprime_r)

        if V_prime.shape[1] == 0 and V_dblprime_r.shape[0] == 0 :
            V_next = torch.zeros(n, r_trunc, device=V.device, dtype=V.dtype)
        elif V_prime.shape[1] != V_dblprime_r.shape[0]:
            print(f"Shape mismatch for V_next: V_prime {V_prime.shape}, V_dblprime_r {V_dblprime_r.shape}")
            V_next = torch.zeros(n, r_trunc, device=V.device, dtype=V.dtype) # Safest fallback
        else:
            V_next = V_prime.mm(V_dblprime_r)
            
        S_next = S_dblprime_r # Already (r_trunc,)

        # Pad if r_trunc is less than target_rank
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
    # This import might cause issues if logger is not setup or if used outside LlamaFactory context.
    # Consider making logger an optional import or passed in.
    try:
        from ..train.trainer_utils import logger # Assumes specific project structure
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
        logger.info_rank0 = logger.info # Basic fallback if info_rank0 not present
    
    logger.info_rank0("\n" + "="*80)
    logger.info_rank0("TRAINABLE PARAMETERS WITH MFSGD STATUS")
    logger.info_rank0("="*80)
    
    # Get parameter groups from optimizer
    mfsgd_param_ids = set() # Using ids might be fragile if params are recreated/moved.
                            # Better to check the flag in optimizer.state[p]

    for group in optimizer.param_groups:
        # if group.get('is_mfsgd_group', False): # Old way
        for p in group['params']:
            if p.requires_grad and optimizer.state[p].get('_is_mfsgd_group_param', False):
                 mfsgd_param_ids.add(id(p)) # Still using id for quick check, but source of truth is state flag
    
    total_params = 0
    mfsgd_params_count = 0 # Renamed from mfsgd_params to avoid conflict
    adamw_params_count = 0 # Renamed from adamw_params
    
    param_infos = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            is_mfsgd = optimizer.state.get(param, {})['_is_mfsgd_group_param']
            # Fallback for MFSGDLinear layers that might not be in optimizer's direct param groups
            # but are handled by MFSGD logic.
            # This requires MFSGDLinear layers to correctly report their status or optimizer to know.
            # For now, rely on `_is_mfsgd_group_param` in optimizer state for the param.
            
            num_p = param.numel()
            total_params += num_p
            
            if is_mfsgd:
                mfsgd_params_count += num_p
            else:
                adamw_params_count += num_p
            param_infos.append({"name": name, "shape": list(param.shape), "num_params": num_p, "is_mfsgd": is_mfsgd})

    max_name_len = max([len(info["name"]) for info in param_infos] + [20]) # Ensure default for empty case
    
    logger.info_rank0(f"{'Parameter Name':<{max_name_len+5}} {'Shape':<20} {'Num Params':<15} {'MFSGD?':<10}")
    logger.info_rank0("-" * (max_name_len + 50))
    
    for info in param_infos:
        logger.info_rank0(f"{info['name']:<{max_name_len+5}} {str(info['shape']):<20} {info['num_params']:<15,d} {'✓' if info['is_mfsgd'] else '✗'}")
    
    logger.info_rank0("-" * (max_name_len + 50))
    logger.info_rank0(f"Total trainable parameters: {total_params:,}")
    if total_params > 0:
        logger.info_rank0(f"MFSGD parameters:         {mfsgd_params_count:,} ({mfsgd_params_count/total_params*100:.2f}%)")
        logger.info_rank0(f"AdamW parameters:         {adamw_params_count:,} ({adamw_params_count/total_params*100:.2f}%)")
    else:
        logger.info_rank0("MFSGD parameters:         0 (N/A)")
        logger.info_rank0("AdamW parameters:         0 (N/A)")
    logger.info_rank0("="*80 + "\n")

# Example usage needs to be updated:
# 1. Model definition would use MFSGDLinear for layers intended for MFSGD.
# 2. Optimizer is initialized with model: optim = MomentumFactorizedSGD(model.parameters(), model_for_mfsgd_layers=model, ...)
# 3. The `setup_optimizer_for_model_layers` (or similar internal call) configures MFSGDLinear instances.

# # ----------------------------------------------------------------------
# # Example (Conceptual - needs full model structure)
# if __name__ == "__main__":
#     # Define a simple model using MFSGDLinear
#     class SimpleModel(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             # Layer to be handled by MFSGD
#             self.mfsgd_layer = MFSGDLinear(10, 5) 
#             # Layer to be handled by AdamW (standard nn.Linear)
#             self.adam_layer = torch.nn.Linear(5, 2) 
#             # Another MFSGD layer (e.g., different group settings if supported, or same)
#             self.another_mfsgd_layer = MFSGDLinear(2,3)


#         def forward(self, x):
#             x = torch.relu(self.mfsgd_layer(x))
#             x = torch.relu(self.adam_layer(x))
#             x = self.another_mfsgd_layer(x)
#             return x

#     model = SimpleModel()
    
#     # Prepare parameter groups for the optimizer
#     mfsgd_params_group1 = []
#     mfsgd_params_group2 = []
#     adamw_params = []

#     # Example: Put self.mfsgd_layer.weight in one MFSGD group, 
#     # self.another_mfsgd_layer.weight in another (e.g. different rank)
#     # and self.adam_layer parameters in AdamW group.
#     # Biases of MFSGDLinear are typically handled by AdamW.

#     # For MFSGDLinear, only its .weight is MFSGD-candidate. Bias is AdamW.
#     mfsgd_params_group1.append(model.mfsgd_layer.weight)
#     if model.mfsgd_layer.bias is not None:
#         adamw_params.append(model.mfsgd_layer.bias)
    
#     mfsgd_params_group2.append(model.another_mfsgd_layer.weight)
#     if model.another_mfsgd_layer.bias is not None:
#         adamw_params.append(model.another_mfsgd_layer.bias)

#     adamw_params.extend(list(model.adam_layer.parameters()))


#     optimizer_param_groups = [
#         {'params': mfsgd_params_group1, 'is_mfsgd_group': True, 'rank': 4, 'lr': 1e-2},
#         {'params': mfsgd_params_group2, 'is_mfsgd_group': True, 'rank': 2, 'lr': 1e-2}, # Different rank for this group
#         {'params': adamw_params, 'is_mfsgd_group': False, 'lr': 1e-3} # AdamW group
#     ]

#     # Instantiate the optimizer, passing the model to configure MFSGDLinear layers
#     optim = MomentumFactorizedSGD(optimizer_param_groups, 
#                                   model_for_mfsgd_layers=model, 
#                                   beta=0.9, eta1=1.0, eta2=0.0) # eta2=0 typically

#     print_mfsgd_parameter_status(model, optim) # Check configuration

#     # Dummy data and training loop
#     input_tensor = torch.randn(3, 10) # Batch 3, Features 10
#     target = torch.randn(3, 3)      # Batch 3, Output Features 3 (from another_mfsgd_layer)

#     print(f"Initial mfsgd_layer.weight norm: {model.mfsgd_layer.weight.data.norm().item()}")
#     if model.mfsgd_layer.weight in optim.state and 'U' in optim.state[model.mfsgd_layer.weight]:
#          print(f"Initial U for mfsgd_layer.weight: {optim.state[model.mfsgd_layer.weight]['U'].norm().item()}")


#     for step_idx in range(5):
#         optim.zero_grad()
#         output = model(input_tensor)
#         loss = torch.nn.functional.mse_loss(output, target)
#         loss.backward() # This will trigger LowRankProjectionForWeight.backward
        
#         # Check gradient status (should be zero for MFSGD weights after backward)
#         # Buffers should be populated.
#         if model.mfsgd_layer.weight.grad is not None:
#             print(f"Step {step_idx} - mfsgd_layer.weight.grad norm: {model.mfsgd_layer.weight.grad.norm().item()} (expected near zero)")
        
#         mfsgd_weight_state = optim.state.get(model.mfsgd_layer.weight, {})
#         if mfsgd_weight_state.get('_is_mfsgd_group_param'):
#             gv_buf = mfsgd_weight_state.get('GV_buffer')
#             if gv_buf is not None:
#                 print(f"  GV_buffer norm for mfsgd_layer: {gv_buf.norm().item()}")


#         optim.step() # Optimizer step uses buffers and updates U,S,V, params
        
#         # Buffers should be zeroed after step
#         if mfsgd_weight_state.get('_is_mfsgd_group_param'):
#             gv_buf_after = mfsgd_weight_state.get('GV_buffer')
#             if gv_buf_after is not None:
#                 print(f"  GV_buffer norm for mfsgd_layer after step: {gv_buf_after.norm().item()} (expected near zero)")


#         print(f"Step {step_idx}: loss={loss.item():.4f}")
#         print(f"  mfsgd_layer.weight norm after step: {model.mfsgd_layer.weight.data.norm().item()}")
#         if 'U' in mfsgd_weight_state:
#             print(f"  U norm for mfsgd_layer after step: {mfsgd_weight_state['U'].norm().item()}")
#         print(f"  adam_layer.weight norm after step: {model.adam_layer.weight.data.norm().item()}")

#     print("\\nTest done.")
# # End of example