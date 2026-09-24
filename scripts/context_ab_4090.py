"""Small matched teacher-history / generated-history flow-loss experiment.

This is an exploratory LoRA comparison, not DMD or a LiveAct training run.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.rollout_probe_4090 import install_lora, make_kv_cache, renoise


def flow_training_pair(clean: torch.Tensor, noise: torch.Tensor, sigma: float):
    if not 0.0 <= sigma <= 1.0:
        raise ValueError('sigma must be in [0, 1]')
    return (1.0 - sigma) * clean + sigma * noise, noise - clean


def paired_step_seed(seed: int, step: int) -> int:
    return seed * 1000003 + step * 7919 + 17


def transition_rmse(video: torch.Tensor, frames_per_block: int) -> dict:
    if video.ndim != 5 or frames_per_block < 1:
        raise ValueError('video must be [B,C,T,H,W] with positive block size')
    frame_rmse = (video[:, :, 1:].float() - video[:, :, :-1].float()).square()
    frame_rmse = frame_rmse.mean(dim=(0, 1, 3, 4)).sqrt().tolist()
    boundary = [v for i, v in enumerate(frame_rmse) if (i + 1) % frames_per_block == 0]
    within = [v for i, v in enumerate(frame_rmse) if (i + 1) % frames_per_block != 0]
    return dict(boundary_rmse=sum(boundary) / len(boundary) if boundary else float('nan'),
                within_rmse=sum(within) / len(within) if within else float('nan'),
                boundary_values=boundary)


def load_latent(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    latent = payload['latent']
    if latent.ndim != 4 or tuple(latent.shape) != (16, 21, 104, 60):
        raise ValueError(f'expected [16,21,104,60] latent in {path}; got {tuple(latent.shape)}')
    return latent.unsqueeze(0).to(device='cuda', dtype=torch.bfloat16)


def load_model(args):
    from wan.modules.causal_model import CausalWanModel

    model = CausalWanModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True).to('cuda').eval()
    if args.ode_checkpoint != 'none':
        state = torch.load(args.ode_checkpoint, map_location='cpu', weights_only=True)
        if 'generator' in state:
            state = state['generator']
        elif 'model' in state and isinstance(state['model'], dict):
            state = state['model']
        if any(key.startswith('model.') for key in state):
            state = {key.removeprefix('model.'): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        del state
    torch.manual_seed(args.seed)
    params = install_lora(model, args.rank)
    return model, params


def make_caches(model, total_frames: int, h: int, w: int):
    tokens_per_frame = h * w // 4
    kv = make_kv_cache(len(model.blocks), 1, total_frames * tokens_per_frame,
                       model.num_heads, model.dim // model.num_heads,
                       torch.bfloat16, 'cuda')
    cross = [dict(is_init=False) for _ in model.blocks]
    return kv, cross, tokens_per_frame


def _noise(shape, generator):
    return torch.randn(shape, device='cuda', dtype=torch.bfloat16, generator=generator)


def fill_history(model, prompt, clean_video, mode, context_sigma, denoise_sigmas,
                 rollout_rng, context_rng, kv, cross, tokens_per_frame):
    history = []
    frames = 3
    h, w = clean_video.shape[-2:]
    seq_len = frames * tokens_per_frame
    context_t = torch.full((1, frames), round(context_sigma * 1000),
                           device='cuda', dtype=torch.long)
    for block in range(6):
        current_start = block * seq_len
        if mode == 'teacher':
            x0 = clean_video[:, :, block * frames:(block + 1) * frames]
        else:
            z = _noise((1, 16, frames, h, w), rollout_rng)
            for index, sigma in enumerate(denoise_sigmas):
                t = torch.full((1, frames), round(sigma * 1000),
                               device='cuda', dtype=torch.long)
                flow = model(z, t=t, context=prompt, seq_len=seq_len,
                             kv_cache=kv, crossattn_cache=cross,
                             current_start=current_start)
                x0 = z - sigma * flow
                if index + 1 < len(denoise_sigmas):
                    z, _ = flow_training_pair(x0, _noise(x0.shape, rollout_rng),
                                              denoise_sigmas[index + 1])
        history.append(x0.detach())
        cached = renoise(x0, _noise(x0.shape, context_rng), context_sigma)
        model(cached, t=context_t, context=prompt, seq_len=seq_len,
              kv_cache=kv, crossattn_cache=cross,
              current_start=current_start)
    return torch.cat(history, dim=2)


def one_example(model, prompt, clean_video, mode, args, step_seed: int,
                training: bool):
    _, _, _, h, w = clean_video.shape
    kv, cross, tokens_per_frame = make_caches(model, 21, h, w)
    rollout_rng = torch.Generator(device='cuda').manual_seed(step_seed + 10000)
    context_rng = torch.Generator(device='cuda').manual_seed(step_seed + 30000)
    current_rng = torch.Generator(device='cuda').manual_seed(step_seed + 20000)
    with torch.no_grad():
        history = fill_history(model, prompt, clean_video, mode,
                               args.context_sigma, args.denoise_sigmas,
                               rollout_rng, context_rng, kv, cross, tokens_per_frame)
    clean = clean_video[:, :, 18:21]
    noise = _noise(clean.shape, current_rng)
    xt, target = flow_training_pair(clean, noise, args.train_sigma)
    t = torch.full((1, 3), round(args.train_sigma * 1000),
                   device='cuda', dtype=torch.long)
    forward_context = torch.enable_grad() if training else torch.no_grad()
    with forward_context:
        prediction = model(xt, t=t, context=prompt,
                           seq_len=3 * tokens_per_frame,
                           kv_cache=kv, crossattn_cache=cross,
                           current_start=18 * tokens_per_frame)
        loss = F.mse_loss(prediction.float(), target.float())
    return loss, history


def evaluate(model, prompt, validation, args, seed: int):
    records = []
    model.eval()
    with torch.no_grad():
        for idx, clean in enumerate(validation):
            seed_i = paired_step_seed(seed + 1000, idx)
            teacher_loss, _ = one_example(model, prompt, clean, 'teacher',
                                          args, seed_i, training=False)
            self_loss, self_history = one_example(model, prompt, clean, 'self',
                                                   args, seed_i, training=False)
            records.append(dict(teacher_flow_mse=teacher_loss.item(),
                                self_flow_mse=self_loss.item(),
                                history_transition=transition_rmse(self_history, 3)))
    return records


def train_mode(model, params, initial, prompt, training, validation, args, mode):
    for p, init in zip(params, initial):
        p.data.copy_(init)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    before = evaluate(model, prompt, validation, args, args.seed)
    losses = []
    start = time.perf_counter()
    for step in range(args.steps):
        step_seed = paired_step_seed(args.seed, step)
        sample_idx = random.Random(step_seed).randrange(len(training))
        optimizer.zero_grad(set_to_none=True)
        loss, _ = one_example(model, prompt, training[sample_idx], mode,
                              args, step_seed, training=True)
        loss.backward()
        if not math.isfinite(loss.item()):
            raise RuntimeError(f'{mode} nonfinite loss at step {step}')
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % args.log_every == 0:
            print(json.dumps(dict(mode=mode, step=step + 1,
                                  flow_mse=losses[-1])), flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    after = evaluate(model, prompt, validation, args, args.seed)
    return dict(mode=mode, train_loss=losses, val_before=before,
                val_after=after, train_seconds=train_seconds,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/data/yg/migrated/home/yg/models/wan2.1-t2v-1.3b')
    parser.add_argument('--ode-checkpoint', default='checkpoints/ode_init.pt')
    parser.add_argument('--data-dir', default='/tmp/wan-liveact-latents')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train-sigma', type=float, default=0.75)
    parser.add_argument('--context-sigma', type=float, default=0.1)
    parser.add_argument('--denoise-sigmas', type=float, nargs='+', default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument('--log-every', type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1 or not all(0 < x <= 1 for x in args.denoise_sigmas):
        raise ValueError('invalid steps or denoising sigmas')
    root = Path(args.data_dir)
    prompt_payload = torch.load(root / 'prompt.pt', map_location='cpu', weights_only=True)
    prompt = [prompt_payload['embedding'].to(device='cuda', dtype=torch.bfloat16)]
    training = [load_latent(root / f'image{i}-start{start}.pt')
                for i in (2, 3) for start in (0, 81)]
    validation = [load_latent(root / f'image4-start{start}.pt') for start in (0, 81)]
    model, params = load_model(args)
    initial = [p.detach().clone() for p in params]
    result = dict(config=vars(args), prompt=prompt_payload['prompt'],
                  train_samples=4, validation_samples=2,
                  model=('official Self-Forcing causal Wan 1.3B ODE-init'
                         if args.ode_checkpoint != 'none' else
                         'untrained causal Wan with bidirectional base weights (code smoke only)'),
                  comparison='matched one-step flow supervision; no DMD')
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for mode in ('teacher', 'self'):
        torch.cuda.reset_peak_memory_stats()
        result[mode] = train_mode(model, params, initial, prompt,
                                  training, validation, args, mode)
        destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({mode: dict(last_train_loss=result[mode]['train_loss'][-1],
                                 val_after=result[mode]['val_after'],
                                 train_seconds=result[mode]['train_seconds'],
                                 peak_reserved_gib=result[mode]['peak_reserved_gib'])
                      for mode in ('teacher', 'self')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
