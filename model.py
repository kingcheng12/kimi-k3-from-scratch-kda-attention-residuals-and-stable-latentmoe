"""
Kimi K3 from Scratch: KDA, Attention Residuals, and Stable LatentMoE

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - short_conv
def short_conv(x, w):
    """Causal depthwise conv: y[t,c] = sum_j w[j,c] * x[t-(K-1)+j, c].

    x: (T, d) sequence.  w: (K, d) per-channel kernel, w[K-1] = current token.
    Positions before the sequence start count as zeros.
    """
    # TODO: accumulate each kernel tap with a shifted slice add

    T, d = x.shape
    K, _ = w.shape

    y = x * 0

    for j in range(K):
        shift = K - 1 - j

        if shift == 0:
            y += x * w[j]
        elif shift < T:
            y[shift:] += x[:T - shift] * w[j]

    return y

# Step 2 - kda_qkv
def swish(x):
    return x / (1 + np.exp(-x))

def kda_qkv(x, params):
    """KDA projections: q,k = L2Norm(Swish(ShortConv(W x))), v = Swish(ShortConv(Wv x)).

    params: dict with Wq (d,dk), Wk (d,dk), Wv (d,dv), cq (K,dk), ck (K,dk), cv (K,dv).
    Returns (q, k, v).  L2Norm divides each row by sqrt(sum(row**2) + 1e-6).
    """
    # TODO: project -> short_conv -> swish, then L2-normalize q and k rows
    q = x @ params["Wq"]
    k = x @ params["Wk"]
    v = x @ params["Wv"]

    # Short convolution
    q = short_conv(q, params["cq"])
    k = short_conv(k, params["ck"])
    v = short_conv(v, params["cv"])

    # Swish
    q = q / (1 + np.exp(-q))
    k = k / (1 + np.exp(-k))
    v = v / (1 + np.exp(-v))

    # L2 normalize q and k
    q = q / np.sqrt(np.sum(q ** 2, axis=-1, keepdims=True) + 1e-6)
    k = k / np.sqrt(np.sum(k ** 2, axis=-1, keepdims=True) + 1e-6)

    return q, k, v

# Step 3 - kda_gates
def kda_gates(x, params):
    """Return (beta, z): write strength sigmoid(x@wb+bb), decay logits x@Wd1@Wd2+ba.

    params: wb (d,), bb scalar, Wd1 (d,r), Wd2 (r,dk), ba (dk,).
    beta: (T,) in (0,1).  z: (T, dk), unbounded.
    """
    beta_logits = x @ params["wb"] + params["bb"]   # (T,)
    beta = 1 / (1 + np.exp(-beta_logits))           # (T,)

    # Decay logits
    z = x @ params["Wd1"] @ params["Wd2"] + params["ba"]  # (T, dk)

    return beta, z

# Step 4 - lower_bounded_decay
def lower_bounded_decay(z, A, g_min=-5.0):
    """alpha = exp(g_min * sigmoid(exp(A) * z)), each entry in [exp(g_min), 1).

    z: (T, dk) decay logits.  A: scalar per-head log-scale.
    """
    scaled_z = np.exp(A) * z

    sigmoid_z = 1 / (1 + np.exp(-scaled_z))

    alpha = np.exp(g_min * sigmoid_z)

    return alpha

# Step 5 - kda_state_update
def kda_state_update(S, k, v, alpha, beta):
    """One KDA step: (I - beta k k^T) @ diag(alpha) @ S + beta * outer(k, v).

    S: (dk, dv).  k: (dk,).  v: (dv,).  alpha: (dk,).  beta: scalar.
    """
    # TODO: decay, then delta-rule erase, then write
    # 1. Decay rows of S
    S_decay = alpha[:, None] * S   

    # 2. Delta-rule erase
    retrieved = k @ S_decay 
    S_erased = S_decay - beta * np.outer(k, retrieved)

    # 3. Write new value
    S_new = S_erased + beta * np.outer(k, v)

    return S_new

# Step 6 - kda_recurrence
def kda_recurrence(q, k, v, alpha, beta, S0=None):
    """Run KDA token by token: update state, then read O[t] = S_t^T q[t].

    Returns (O, S_final) with O of shape (T, dv). S0 defaults to zeros; never
    mutate the caller's S0.
    """
    T, dk = q.shape
    dv = v.shape[1]

    # Initialize state
    if S0 is None:
        S = np.zeros((dk, dv), dtype=q.dtype)
    else:
        S = S0.copy()

    O = np.zeros((T, dv), dtype=q.dtype)

    for t in range(T):
        # 1. Update state using current token
        S = kda_state_update(
            S,
            k[t],          # (dk,)
            v[t],          # (dv,)
            alpha[t],      # (dk,)
            beta[t]        # scalar
        )

        # 2. Read from the UPDATED state
        O[t] = q[t] @ S   # (dk,) @ (dk, dv) -> (dv,)

    return O, S

# Step 7 - cumulative_decay
def cumulative_decay(alpha):
    """Inclusive channel-wise cumulative product of alpha down the time axis.

    alpha: (C, dk) per-step retention factors -> Gamma: (C, dk).
    """
    return np.cumprod(alpha, axis=0)

# Step 8 - chunk_pseudo_values (not yet solved)
# TODO: implement

# Step 9 - kda_chunkwise (not yet solved)
# TODO: implement

# Step 10 - kda_output_gate (not yet solved)
# TODO: implement

# Step 11 - mla_compress_reconstruct (not yet solved)
# TODO: implement

# Step 12 - nope_attention (not yet solved)
# TODO: implement

# Step 13 - mla_output_gate (not yet solved)
# TODO: implement

# Step 14 - hybrid_schedule (not yet solved)
# TODO: implement

# Step 15 - attnres_weights (not yet solved)
# TODO: implement

# Step 16 - attnres_full (not yet solved)
# TODO: implement

# Step 17 - block_partial_sums (not yet solved)
# TODO: implement

# Step 18 - attnres_block (not yet solved)
# TODO: implement

# Step 19 - situ_glu (not yet solved)
# TODO: implement

# Step 20 - route_topk (not yet solved)
# TODO: implement

# Step 21 - routed_experts (not yet solved)
# TODO: implement

# Step 22 - stable_latent_moe (not yet solved)
# TODO: implement

# Step 23 - topk_cutoffs (not yet solved)
# TODO: implement

# Step 24 - quantile_balance_update (not yet solved)
# TODO: implement

# Step 25 - histogram_quantile (not yet solved)
# TODO: implement

# Step 26 - newton_schulz (not yet solved)
# TODO: implement

# Step 27 - per_head_muon (not yet solved)
# TODO: implement

# Step 28 - mini_k3_forward (not yet solved)
# TODO: implement

