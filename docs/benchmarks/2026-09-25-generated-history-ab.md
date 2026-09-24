# Preregistered single-4090 context training comparison

## Question and limit

Does brief LoRA training with a generator's own detached history improve held-out **self-history flow prediction** relative to the same training under ground-truth history? This tests a train/inference context mismatch with a real 1.3B causal model. It does not reproduce DMD, Vidu S2 SRF, or train SoulX-LiveAct. A lower flow loss is not enough to claim improved video quality, motion or lip-sync.

## Fixed data and model

- Causal Wan 1.3B with the official released ODE-initialization checkpoint. Base frozen; rank-4 LoRA on self-attention Q and V only, same initialization in both arms.
- One real UMT5-XXL encoded prompt: “A person faces the camera, speaks naturally, and gestures with the hands.” Its 18×4096 embedding is cached once before generator training.
- Source clips are **existing LiveAct generated videos**, not captured real-person footage. Images/audio 2 and 3 form training identities; image/audio 4 is held out. The first and second nonoverlapping 81-frame windows of each clip yield 21 Wan latent frames each. All are resized from 416×720 to 480×832, preserving portrait aspect ratio. There are four train windows and two held-out windows; this is a small pilot, not a representative dataset.
- Wan VAE and T5 remain frozen. Their outputs are cached in `/tmp/wan-liveact-latents` and are excluded from Git. The video source SHA-256 and latent shape are embedded in each latent file.

## Paired arms

For each 21-latent-frame window, blocks 0–5 contain three frames each and form context. In **teacher history**, the cache is filled from the encoded target video; in **self history**, the causal generator rolls out those six blocks from noise with four decreasing denoising sigmas `[1.0, 0.75, 0.5, 0.25]`. Both arms use the same context re-noising sigma `0.1`. History is detached. The final three-frame target is noised at sigma `0.75`; trainable generator output is supervised by the flow target `noise − clean_latent` with MSE. This is ordinary flow matching, not distribution-matching distillation.

Both arms use the same video index schedule, final-block noise seeds, context-noise seeds, optimizer, step count and starting adapter weights. Rollout noise exists only in the self-history arm. Each model is evaluated on both teacher and self contexts before and after training. The **primary** comparison is mean held-out self-context flow MSE after training; the before values verify matched initialization. Secondary diagnostics are the teacher-context MSE and latent cross-block versus within-block transition RMSE of the self-generated history. The latter can detect a gross seam but is not a perceptual motion score.

## Decision rule

Run at least three adapter seeds if the one-step smoke run fits. A credible directional signal requires a lower self-context MSE for the self-history arm in all or most seeds **without** a clear rise in its teacher-context MSE or latent seam ratio. If results are mixed, label the method inconclusive. No result from this small Wan pilot enters the LiveAct PR or resume as a solved forcing contribution. Visual and lip-sync evidence on actual LiveAct rollouts would still be required for such a claim.

## Reproduce

```bash
/home/yg/miniforge3/envs/digithuman/bin/python -m scripts.prepare_liveact_wan_latents \
  --video /tmp/liveact-rollout-image2-8s.mp4 \
  --output /tmp/wan-liveact-latents/image2-start0.pt
/home/yg/miniforge3/envs/digithuman/bin/python -m scripts.prepare_wan_text
/home/yg/miniforge3/envs/digithuman/bin/python -m scripts.context_ab_4090 \
  --steps 20 --seed 0 --output docs/benchmarks/artifacts/context-ab-seed0.json
```

Repeat video encoding for identities 2, 3 and 4 at starts 0 and 81, then repeat the training command at seeds 1 and 2. Exact commands, environment and outputs will be recorded with the measurements.
