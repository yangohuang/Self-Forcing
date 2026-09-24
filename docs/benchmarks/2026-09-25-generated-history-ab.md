# Preregistered single-4090 context training comparison

## Question and limit

Does brief LoRA training with a generator's own detached history improve held-out **self-history flow prediction** relative to the same training under ground-truth history? This tests a train/inference context mismatch with a real 1.3B causal model. It does not reproduce DMD, Vidu S2 SRF, or train SoulX-LiveAct. A lower flow loss is not enough to claim improved video quality, motion or lip-sync.

There is a deliberate but severe proxy limitation: a text-only student may generate a valid past that does not match the single LiveAct target video, especially because no reference image or audio is supplied to Wan. Direct flow supervision of that target can then penalize plausible alternatives. A negative or mixed result cannot refute self-forcing or SRF; it can only assess this small supervised proxy and guide the next experiment.

## Fixed data and model

- Causal Wan 1.3B with the official released ODE-initialization checkpoint. Base frozen; rank-4 LoRA on self-attention Q and V only, same initialization in both arms.
- One real UMT5-XXL encoded prompt: “A person faces the camera, speaks naturally, and gestures with the hands.” Its 18×4096 embedding is cached once before generator training.
- Source clips are **existing LiveAct generated videos**, not captured real-person footage. Images/audio 2 and 3 form training identities; image/audio 4 is held out. The first and second nonoverlapping 81-frame windows of each clip yield 21 Wan latent frames each. All are resized from 416×720 to 480×832, preserving portrait aspect ratio. There are four train windows and two held-out windows; this is a small pilot, not a representative dataset.
- Wan VAE and T5 remain frozen. Their outputs are cached in `/tmp/wan-liveact-latents` and are excluded from Git. The video source SHA-256 and latent shape are embedded in each latent file.
- A second held-out read of image/audio 4 encodes 193 video frames into 49 latents and uses the first **48 latent frames** for a longer-than-training check. It shares source identity and frames with the two 21-frame validation windows, so it is a horizon extension, not an independent third example. The causal model uses a 21-latent-frame rolling KV window for this check.

## Paired arms

For each 21-latent-frame window, blocks 0–5 contain three frames each and form context. In **teacher history**, the cache is filled from the encoded target video; in **self history**, the causal generator rolls out those six blocks from noise with four decreasing sigmas `[1.0, 0.9375, 0.8333, 0.625]`. These are the shift-5 values corresponding to the official nominal positions `[1000, 750, 500, 250]`. Both arms use the same context re-noising sigma `0.1`. History is detached. The final three-frame target is noised at sigma `0.75`; trainable generator output is supervised by the flow target `noise − clean_latent` with MSE. This is ordinary flow matching, not distribution-matching distillation.

Both arms use the same video index schedule, final-block noise seeds, context-noise seeds, optimizer, step count and starting adapter weights. Rollout noise exists only in the self-history arm. Each model is evaluated on both teacher and self contexts before and after training. The **primary** comparison is mean held-out self-context flow MSE after training; the before values verify matched initialization. Secondary diagnostics are the teacher-context MSE and latent cross-block versus within-block transition RMSE of the self-generated history. The latter can detect a gross seam but is not a perceptual motion score.

The 48-frame horizon uses the same metrics with 15 generated-history blocks and a final three-frame flow target. A rolling KV cache holds at most 21 frames; that tests error accumulation under a bounded history. It does not demonstrate indefinite generation, reference consistency or audio synchronization.

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

## Results: three preregistered seeds

The released ODE-init checkpoint was downloaded and verified against the publisher's SHA-256 `b5396b8076ab3b920c9e4f4a2b52daa2c98c9983fb5e067ae5160fdf778dce21`. Each seed ran 20 AdamW updates for each mode at learning rate `1e-4`, rank 4, batch 1, train sigma `0.75`, and context sigma `0.1`. The source videos were generated previously by LiveAct under seed 42. Training and validation used a real UMT5 embedding; **Wan received no image or audio condition**. Short validation had two 21-latent-frame windows of held-out identity 4; the 48-frame check used its same video. Every mode and seed completed on the RTX 4090; peak CUDA reserved memory was **18.635 GiB**.

The table reports flow MSE after training. “Δ” is *self-history-trained minus target-history-trained*, so negative is favorable on that column. Raw per-window measurements and all before values are preserved in [`seed0`](artifacts/context-ab-seed0.json), [`seed1`](artifacts/context-ab-seed1.json), and [`seed2`](artifacts/context-ab-seed2.json).

| Seed | 21f self-context: target / self | Δ self-context | Δ target-context | 48f self-context: target / self | Δ 48f self-context | 48f latent seam ratio: target / self |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.170499 / 0.170052 | −0.000447 | +0.000428 | 0.182286 / 0.182452 | +0.000166 | 1.235 / 1.234 |
| 1 | 0.169373 / 0.168943 | −0.000430 | +0.000459 | 0.174412 / 0.173681 | −0.000731 | 1.290 / 1.300 |
| 2 | 0.171625 / 0.171271 | −0.000354 | +0.000424 | 0.183312 / 0.182828 | −0.000485 | 1.215 / 1.192 |

The two arms' pretraining validation values matched exactly in each seed. Before training, the same ODE-initialized model had held-out self-history MSE **2.72–2.89×** its target-history MSE. This is a context/target mismatch on these clips; it is not proof that autoregressive error alone caused the difference because Wan lacks the input image and audio that specify the particular future. After training, the short-window self-context improvement averaged only **0.000410 MSE** (about 0.24% of the target-history-trained value), while target-context MSE rose by **0.000437** in all three seeds. The 48-frame self-context delta was mixed (mean −0.000350, range −0.000731 to +0.000166); its latent seam ratio was also mixed. Training time per seed was about **49.4 s** with target history versus **166.2 s** with generated history, excluding evaluation.

**Decision:** the preregistered promotion gate fails. The small self-context MSE gain trades against target-context error and does not consistently persist at the longer horizon or improve the seam diagnostic. No video-quality, identity, lip-sync, LiveAct continuity or SRF claim follows. We retain this as a measured single-card training experiment and a reason to avoid an upstream quality PR from the current proxy.

The next method experiment would need an aligned reference/audio-conditioned generator or a distributional teacher objective, rather than forcing a text-only generated trajectory to reconstruct one particular audio-driven LiveAct future. Vidu S2's [Self-Replay Forcing](https://arxiv.org/html/2609.11638) replays the re-noised trajectory with gradient connections across blocks and applies DMD/perceptual supervision. This pilot detaches the generated history and differentiates only the final block with ordinary flow loss, so it deliberately does **not** implement that algorithm.

## Paired visible rollout

The first seed was rerun with the same protocol solely to save both adapter weights for rendering; small floating-point run-to-run variation was observed, so this artifact is not substituted for the three-seed table. Both trained adapters then generated 48 latents from the **same prompt and seed 42** with the four-step schedule and a 21-frame rolling cache. Wan VAE decoded each to 189 silent frames at 480×832, 24 fps (7.875 s). Both MP4 files were checked with FFprobe. The [contact sheet](artifacts/context-ab-seed0-contact.png) samples each second; source files are [target-history training](artifacts/context-ab-seed0-target-history.mp4) and [generated-history training](artifacts/context-ab-seed0-generated-history.mp4).

In this one matched sample, both outputs lose face and scene clarity later in the rollout, and the generated-history-trained version shows no clear visual recovery. The free-rollout latent seam/within RMSE ratio is **1.219** for target-history training and **1.243** for generated-history training. This single visual seed supports the conservative no-promotion decision; it is not a multi-seed perceptual score. These Wan videos do not carry LiveAct's original image/audio conditioning and cannot answer whether LiveAct's frame-373 jump is fixed.
