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

# Step 8 - chunk_pseudo_values
def chunk_pseudo_values(k, v, alpha, beta, S0):
    """Solve (I + diag(beta) strict_tril(Khat Kcheck^T)) U = diag(beta)(V - Khat S0).

    Khat = k * Gamma, Kcheck = k / Gamma, Gamma = cumulative_decay(alpha).
    Returns U of shape (C, dv).
    """
    C = k.shape[0]

    Gamma = cumulative_decay(alpha)

    Khat = k * Gamma
    Kcheck = k / Gamma

    # Pairwise interaction matrix
    interaction = Khat @ Kcheck.T

    # Keep only entries strictly below the diagonal
    L = np.tril(interaction, k=-1)

    # I + diag(beta) @ L
    # Multiplying by diag(beta) scales each ROW of L.
    A = np.eye(C, dtype=k.dtype) + beta[:, None] * L

    # diag(beta) @ (V - Khat @ S0)
    residual = v - Khat @ S0      
    rhs = beta[:, None] * residual 

    # Solve A @ U = rhs
    U = np.linalg.solve(A, rhs)

    return U

# Step 9 - kda_chunkwise
def kda_chunkwise(q, k, v, alpha, beta, chunk_size, S0=None):
    """Chunkwise-parallel KDA (Eq. 4): O_c = Qhat @ S + tril(Qhat Kcheck^T) @ U.

    State hand-off: S <- Gamma[-1][:,None] * (S + Kcheck^T U). Must equal
    kda_recurrence for every chunk size. Returns (O, S_final).
    """
    T, dk = q.shape
    dv = v.shape[1]

    # Initial state
    if S0 is None:
        S = np.zeros((dk, dv), dtype=q.dtype)
    else:
        S = S0.copy()

    O = np.zeros((T, dv), dtype=q.dtype)

    # Process one chunk at a time
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)

        qc = q[start:end]    
        kc = k[start:end]      
        vc = v[start:end]      
        ac = alpha[start:end]     
        bc = beta[start:end]   

        # Channel-wise cumulative decay
        Gamma = cumulative_decay(ac)    

        # Transformed Q/K
        Qhat = qc * Gamma              
        Kcheck = kc / Gamma        

        # Compute pseudo-values using state at chunk entrance
        U = chunk_pseudo_values(kc, vc, ac, bc, S)                              
        # Inclusive causal interactions
        causal = np.tril(Qhat @ Kcheck.T) 

        # Chunk outputs
        O[start:end] = (Qhat @ S        + causal @ U      )

        # Hand state to next chunk
        S = Gamma[-1][:, None] * (S + Kcheck.T @ U)

    return O, S

# Step 10 - kda_output_gate
def kda_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * RMSNorm(o)) @ Wo, RMSNorm = o / sqrt(mean(o^2)+1e-6).

    o: (T, dv) recurrent outputs.  x: (T, d) layer input.  Returns (T, d).
    """
    rms = np.sqrt(np.mean(o ** 2, axis=-1, keepdims=True) + 1e-6)
    o_norm = o / rms                 

    # Output gate
    gate_logits = x @ Wg                
    gate = 1 / (1 + np.exp(-gate_logits)) 

    # Gate recurrent output, then project
    y = (gate * o_norm) @ Wo            

    return y

# Step 11 - mla_compress_reconstruct
def mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, n_heads):
    """c = x @ Wc; K = (c @ Wk_up).reshape(T, H, dh); V likewise.

    Returns (c, K, V) with shapes (T, r), (T, H, dh), (T, H, dh).
    """
    T = x.shape[0]

    # Compress latent representation
    c = x @ Wc                        

    # Reconstruct full K and V
    K_flat = c @ Wk_up           
    V_flat = c @ Wv_up             

    # Infer per-head dimension
    total_dim = K_flat.shape[-1]
    assert total_dim % n_heads == 0

    dh = total_dim // n_heads

    # Split into attention heads
    K = K_flat.reshape(T, n_heads, dh)   
    V = V_flat.reshape(T, n_heads, dh)  

    return c, K, V

# Step 12 - nope_attention
def nope_attention(x, Wq, Wc, Wk_up, Wv_up, n_heads):
    """Causal multi-head attention over MLA-reconstructed K,V - no positions.

    Q = (x @ Wq).reshape(T, H, dh); per head softmax(QK^T/sqrt(dh)) V with a
    causal mask; concatenate heads -> (T, H*dh).
    """
    T = x.shape[0]

    # Reconstruct K, V from compressed latent
    _, K, V = mla_compress_reconstruct(
        x, Wc, Wk_up, Wv_up, n_heads
    )                         

    dh = K.shape[-1]

    # Queries
    Q = (x @ Wq).reshape(T, n_heads, dh)  

    # Put heads first for easier batched attention
    Q = Q.transpose(1, 0, 2)        
    K = K.transpose(1, 0, 2)         
    V = V.transpose(1, 0, 2)         

    # Attention scores
    scores = Q @ K.transpose(0, 2, 1)
    scores = scores / np.sqrt(dh)

    # Causal mask: prevent attending to future tokens
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask[None, :, :], -np.inf, scores)

    # Numerically stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    # Weighted sum of values
    out = weights @ V                

    # Back to (T, H, dh), then concatenate heads
    out = out.transpose(1, 0, 2)       
    out = out.reshape(T, n_heads * dh)   

    return out

# Step 13 - mla_output_gate
def mla_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * o) @ Wo - note: no RMSNorm here, unlike KDA's gate.

    o: (T, H*dh) attention output.  x: (T, d) layer input.  Returns (T, d).
    """
    gate_logits = x @ Wg
    gate = 1 / (1 + np.exp(-gate_logits))

    y = (gate * o) @ Wo

    return y

# Step 14 - hybrid_schedule
def hybrid_schedule(n_repeats):
    """['KDA','KDA','KDA','MLA'] repeated n_repeats times, plus a final 'MLA'."""
    return ['KDA', 'KDA', 'KDA', 'MLA'] * n_repeats + ['MLA']

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

