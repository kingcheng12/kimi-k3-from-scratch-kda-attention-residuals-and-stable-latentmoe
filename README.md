# Kimi K3 from Scratch: KDA, Attention Residuals, and Stable LatentMoE

Build every architectural innovation from the Kimi K3 technical report in numpy at toy scale: Kimi Delta Attention with lower-bounded decay and an exact chunkwise-parallel form, Gated MLA with NoPE, Attention Residuals over depth, Stable LatentMoE, and per-head Muon orthogonalization, then assemble a working mini K3 block stack.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** short_conv
- [x] **2.** kda_qkv
- [x] **3.** kda_gates
- [x] **4.** lower_bounded_decay
- [x] **5.** kda_state_update
- [x] **6.** kda_recurrence
- [x] **7.** cumulative_decay
- [x] **8.** chunk_pseudo_values
- [x] **9.** kda_chunkwise
- [x] **10.** kda_output_gate
- [x] **11.** mla_compress_reconstruct
- [x] **12.** nope_attention
- [x] **13.** mla_output_gate
- [x] **14.** hybrid_schedule
- [x] **15.** attnres_weights
- [x] **16.** attnres_full
- [ ] **17.** block_partial_sums
- [ ] **18.** attnres_block
- [ ] **19.** situ_glu
- [ ] **20.** route_topk
- [ ] **21.** routed_experts
- [ ] **22.** stable_latent_moe
- [ ] **23.** topk_cutoffs
- [ ] **24.** quantile_balance_update
- [ ] **25.** histogram_quantile
- [ ] **26.** newton_schulz
- [ ] **27.** per_head_muon
- [ ] **28.** mini_k3_forward

---

Built on Deep-ML.
