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
        # Full SVD (may be expensive for large matrices, but fine for demonstration)
        U_full, S_full, V_full = torch.svd(p)
        r_trunc = min(rank, min(m, n))
        self.U = U_full[:, :r_trunc].clone()
        self.S = S_full[:r_trunc].clone()
        self.V = V_full[:, :r_trunc].clone()


#############################################
#   The MomentumFactorizedSGD Optimizer
#############################################
class MomentumFactorizedSGD(Optimizer):
    r"""
    Implements a rank-r factorized momentum update:
    
      1) We store M_t = U_t diag(S_t) V_t^T  (rank r).
      2) Each step, we do:

         (a) Tangent project the new gradient G_t:
             G_hat = U_t U_t^T G_t
                     + G_t V_t V_t^T
                     - U_t U_t^T G_t V_t V_t^T.

         (b) Update the momentum factor to represent:
             M_t = G_hat + beta * M_{t-1} 
                  = G_hat + beta * (U_t diag(S_t) V_t^T)
             with rank at most 2r, truncated back to r via the
             block-matrix approach:

             [U_t | G_t V_t] = U'_t R_{U_t}, 
             [V_t | G_t^T U_t] = V'_t R_{V_t},     # QR factorizations

             B = [[ beta diag(S_t) - U_t^T G_t V_t,   I_r ],
                  [             I_r,                 0_r ]]

             Mid = R_{U_t} * B * R_{V_t}^T
             => SVD_r(Mid) = U'' diag(S'') V''^T
             => U_{t+1} = U'_t U'',
                S_{t+1} = S'',
                V_{t+1} = V'_t V''.

         (c) Finally, update p:
             p <- p - lr * [
                  eta1 * (U_{t+1} diag(1/S_{t+1}) V_{t+1}^T)
                  + eta2 * (I - P_U) G_t (I - P_V)
             ].

    Args:
      params: iterable of parameters to optimize (all must be 2D)
      lr:     global learning rate
      rank:   integer rank r
      beta:   momentum decay factor
      eta1:   scale factor for the low-rank momentum term
      eta2:   scale factor for the orthogonal complement gradient
      use_current_projection: flag to use current or previous projections
      use_ones_for_nonzero_s: flag to handle singular values
      eps:    epsilon value for numerical stability
      nesterov: flag to use Nesterov momentum
      max_value: maximum value for clipping reciprocal singular values
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
                 # AdamW fallback hyper-parameters
                 adam_betas: tuple = (0.9, 0.999),
                 adam_eps: float = 1e-8,
                 adam_weight_decay: float = 0.0):

        # Basic validations (borrowed from AdamW implementation)
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
            mfsgd_eps = group['mfsgd_eps']
            nesterov = group['nesterov']
            max_value = group['max_value']

            # AdamW fallback hyperparams
            adam_beta1, adam_beta2 = group['adam_betas']
            adam_eps = group['adam_eps']
            adam_weight_decay = group['adam_weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Route parameters: 2-D tensors use MFSGD, others fallback to AdamW
                is_matrix = p.dim() == 2

                # ---------------- MFSGD branch ----------------
                if is_matrix:
                    # Retrieve or init (U, S, V)
                    state = self.state[p]
                    if 'momentum_factor' not in state:
                        state['momentum_factor'] = MomentumFactor(p, rank)

                    mf = state['momentum_factor']
                    U, S, V = mf.U, mf.S, mf.V

                    # Current gradient
                    G_t = p.grad

                    # == (b) Update the momentum factor ==
                    U_next, S_next, V_next = self._update_momentum_factor(U, S, V, G_t, beta)

                    # Store the new factors
                    mf.U = U_next
                    mf.S = S_next
                    mf.V = V_next

                    # If using Nesterov momentum, apply the update again
                    if nesterov:
                        U_next, S_next, V_next = self._update_momentum_factor(U_next, S_next, V_next, G_t, beta)

                    # == (c) Parameter update ==
                    # Removed explicit calculation of U_nextU_nextT and V_nextV_nextT to save memory

                    # Create mask for non-zero singular values
                    non_zero_mask = S_next.abs() > mfsgd_eps
                    safe_reciprocal_S = torch.zeros_like(S_next)

                    if use_ones_for_nonzero_s:
                        safe_reciprocal_S[non_zero_mask] = 1.0
                    else:
                        safe_reciprocal_S[non_zero_mask] = 1.0 / (S_next[non_zero_mask])
                        safe_reciprocal_S = torch.clamp(safe_reciprocal_S, max=max_value)

                    USVt_next = (U_next * safe_reciprocal_S.unsqueeze(0)) @ V_next.T

                    if use_current_projection:
                        # Use U_next and V_next for projections
                        U_proj, V_proj = U_next, V_next
                    else:
                        # Use U and V from the start of the step for projections
                        U_proj, V_proj = U, V

                    # Orthogonal complement calculation without forming full P_U, P_V
                    # left_ortho = G_t - P_U @ G_t = G_t - U_proj @ (U_proj.T @ G_t)
                    UTG = U_proj.t().mm(G_t)
                    UUTG = U_proj.mm(UTG)
                    left_ortho = G_t - UUTG

                    # right_ortho = left_ortho - left_ortho @ P_V = left_ortho - (left_ortho @ V_proj) @ V_proj.T
                    left_ortho_V = left_ortho.mm(V_proj)
                    left_ortho_VV = left_ortho_V.mm(V_proj.t())
                    right_ortho = left_ortho - left_ortho_VV

                    # In-place update
                    p.data.add_(eta1 * USVt_next + eta2 * right_ortho, alpha=-lr)

                else:
                    # ---------------- AdamW fallback branch ----------------
                    state = self.state[p]

                    # State initialization for AdamW moments
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    state['step'] += 1
                    step = state['step']

                    grad = p.grad

                    # Apply weight decay (decoupled)
                    if adam_weight_decay != 0:
                        p.data.mul_(1 - lr * adam_weight_decay)

                    # AdamW moments update
                    exp_avg.mul_(adam_beta1).add_(grad, alpha=1 - adam_beta1)
                    exp_avg_sq.mul_(adam_beta2).addcmul_(grad, grad, value=1 - adam_beta2)

                    bias_correction1 = 1 - adam_beta1 ** step
                    bias_correction2 = 1 - adam_beta2 ** step

                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(adam_eps)
                    step_size = lr / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

    #############################################
    # The core rank-2r update of momentum
    #############################################
    def _update_momentum_factor(self, U, S, V, G, beta):
        """
        Internal method to update (U, S, V) so that
           M_new = G + beta * (U diag(S) V^T)
        is represented by rank-r factors (U', S', V'),
        using the 2-QR + block-matrix + 2r×2r SVD approach.

        Equations match:

          [ U | G V ] = U'  R_U
          [ V | G^T U ] = V' R_V

          B = [[ beta diag(S) - U^T G V,  I_r ],
               [          I_r,           0_r ]]

          Mid = R_U * B * R_V^T   =>  SVD_r(Mid) = U'' diag(S'') V''^T

          =>  U_next = U' * U'',  S_next = S'',  V_next = V' * V''
        """
        m, r = U.shape
        n = V.shape[0]
        assert V.shape[1] == r, "V must be (n x r)"
        assert S.shape[0] == r, "S must be (r,)"
        assert G.shape == (m, n), "G must be (m x n)"

        # 1) row_block = [ U,  G V ]
        GV = G.mm(V)  # (m x r)
        row_block = torch.cat([U, GV], dim=1)  # (m, 2r)
        U_prime, R_U = torch.linalg.qr(row_block, mode='reduced')  # U_prime:(m,2r), R_U:(2r,2r)

        # 2) col_block = [ V,  G^T U ]
        GTU = G.t().mm(U)  # (n, r)
        col_block = torch.cat([V, GTU], dim=1)  # (n, 2r)
        V_prime, R_V = torch.linalg.qr(col_block, mode='reduced')  # V_prime:(n,2r), R_V:(2r,2r)

        # 3) B = block matrix of size (2r,2r)
        beta_Sigma = torch.diag(beta * S)           # (r x r)
        UTGV = U.t().mm(G).mm(V)                    # (r x r)
        top_left = beta_Sigma - UTGV                # (r x r)
        eye_r = torch.eye(r, device=U.device)
        zero_r = torch.zeros(r, r, device=U.device)

        top_row = torch.cat([top_left, eye_r], dim=1)   # (r,2r)
        bot_row = torch.cat([eye_r, zero_r], dim=1)     # (r,2r)
        B = torch.cat([top_row, bot_row], dim=0)        # (2r,2r)

        # 4) Mid = R_U * B * R_V^T
        Mid = R_U.mm(B).mm(R_V.t())  # (2r, 2r)

        # 5) SVD on Mid, truncated to rank r
        U_dblprime, S_dblprime, V_dblprime = torch.svd(Mid)
        U_dblprime_r = U_dblprime[:, :r]  # (2r, r)
        S_dblprime_r = S_dblprime[:r]     # (r,)
        V_dblprime_r = V_dblprime[:, :r]  # (2r, r)

        # 6) Pull back => U_next, S_next, V_next
        U_next = U_prime.mm(U_dblprime_r)  # (m, r)
        S_next = S_dblprime_r.clone()      # (r,)
        V_next = V_prime.mm(V_dblprime_r)  # (n, r)

        return U_next, S_next, V_next


# # ----------------------------------------------------------------------
# # Example usage:
# if __name__ == "__main__":
#     # Suppose a single 2D weight
#     w = torch.nn.Parameter(torch.randn(8, 6, requires_grad=True))

#     # Dummy closure/ loss
#     def loss_fn():
#         # e.g., sum of w for demonstration
#         return w.sum()

#     # Instantiate the optimizer
#     optim = MomentumFactorizedSGD([w], lr=1e-2, rank=2, beta=0.9, eta1=1.0, eta2=1.0)

#     # Simple loop
#     for step in range(5):
#         optim.zero_grad()
#         l = loss_fn()
#         l.backward()
#         optim.step()
#         print(f"Step {step}: loss={l.item():.4f}")