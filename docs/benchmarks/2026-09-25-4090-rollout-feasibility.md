# RTX 4090 generated-history training feasibility

This is a bounded training-memory experiment on the official Self-Forcing causal Wan 1.3B generator. It is **not** the published DMD training recipe, SRF, or a SoulX-LiveAct quality result. The conditioning embedding and regression target are synthetic; only the model weights and autoregressive cache path are real. The intended measurement is whether a detached generated-history block can be followed by a trainable second block on one RTX 4090.

## Setup

- Source: official `guandeh17/Self-Forcing`, commit `33593df` (cloned 2026-09-25).
- Weight source: local Wan 2.1 T2V 1.3B base checkpoint, 825 tensors, `blocks.0.self_attn.q.weight` matches the 1536-dimensional model. It is **not** the released Self-Forcing ODE initialization checkpoint.
- GPU: RTX 4090 24 GiB; environment `/home/yg/miniforge3/envs/digithuman`, PyTorch 2.6.0+cu124, flash-attn 2.7.2, diffusers 0.37.1.
- Method: freeze base; add rank-4 LoRA to causal self-attention Q and V; create first block without gradients; re-noise its detached x0 and rebuild the KV cache; backpropagate the second block's MSE into LoRA parameters. This MSE is a feasibility signal, not a meaningful generative objective.

## Measurements

All rows use batch 1, bf16 base, rank-4 Q/V LoRA (737,280 trainable parameters), one denoising prediction per block, detached and re-noised generated history, and one AdamW step on the last block. `H×W` is **latent** resolution; 60×104 corresponds to the spatial shape in the official training config. Timings below exclude cold model load and CUDA context initialization. Full invocation: `TF_CPP_MIN_LOG_LEVEL=3 /home/yg/miniforge3/envs/digithuman/bin/python -m scripts.rollout_probe_4090 --latent-h 60 --latent-w 104 --frames-per-block 3 --history-blocks 6 --seed 0 --output docs/benchmarks/artifacts/rollout-probe-60x104-f3-7blocks-seed0.json`.

| Latent size | History + train block | Seed | History time | Train forward + backward | Peak CUDA reserved | Loss | Gradient norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16×24 | 1×1 + 1×1 frames | 0 | 0.350 s | 0.101 s | 3.08 GiB | 1.1130 | 0.01487 |
| 32×48 | 1×3 + 1×3 frames | 0 | 0.249 s | 0.134 s | 5.93 GiB | 1.0584 | 0.00774 |
| 60×104 | 1×3 + 1×3 frames | 0 | 0.523 s | 0.561 s | 14.41 GiB | 1.0347 | 0.00586 |
| 60×104 | 6×3 + 1×3 frames | 0 | 3.073 s | 1.016 s | 18.68 GiB | 1.0525 | 0.00295 |
| 60×104 | 6×3 + 1×3 frames | 1 | 3.078 s | 1.017 s | 18.68 GiB | 1.0431 | 0.00358 |

Every run reported finite loss, finite nonzero LoRA gradients and successful optimizer completion. The 21-frame trials reserved 18.68 GiB of the card's 24 GiB. The last-block timing is a one-step backward probe, not end-to-end training throughput: official DMD requires other networks, text encoding, multiple denoising steps and validation. The initial weights are a **bidirectional Wan base transplanted into a causal architecture**, not the trained ODE-init generator. The random conditioning and random MSE target make the scalar loss unsuitable for video-quality comparisons.

The next gate is a real pretrained causal checkpoint and a controlled clean-history versus generated-history comparison with held-out videos. A successful optimizer step alone does not show that forcing improves motion continuity or the SoulX block-12 discontinuity.
