"""Render a paired trained adapter for visual inspection; not a quality metric."""

import argparse
import gc
import json
from pathlib import Path

import cv2
import torch

from scripts.context_ab_4090 import load_model, make_caches, transition_rmse
from scripts.rollout_probe_4090 import renoise


def generate(args):
    model, params = load_model(args)
    adapter = torch.load(args.adapter, map_location='cpu', weights_only=True)
    if adapter['rank'] != args.rank or len(adapter['parameters']) != len(params):
        raise ValueError('adapter shape or rank does not match the model')
    for parameter, value in zip(params, adapter['parameters']):
        if parameter.shape != value.shape:
            raise ValueError('adapter tensor shape mismatch')
        parameter.data.copy_(value)
    prompt_payload = torch.load(args.prompt_embedding, map_location='cpu', weights_only=True)
    prompt = [prompt_payload['embedding'].to('cuda', dtype=torch.bfloat16)]
    blocks = args.latent_frames // 3
    kv, cross, tokens_per_frame = make_caches(model, args.latent_frames, 104, 60)
    seq_len = 3 * tokens_per_frame
    rollout_rng = torch.Generator(device='cuda').manual_seed(args.seed + 40000)
    context_rng = torch.Generator(device='cuda').manual_seed(args.seed + 50000)
    context_t = torch.full((1, 3), round(args.context_sigma * 1000),
                           device='cuda', dtype=torch.long)
    chunks = []
    with torch.no_grad():
        for block in range(blocks):
            z = torch.randn((1, 16, 3, 104, 60), device='cuda',
                            dtype=torch.bfloat16, generator=rollout_rng)
            start = block * seq_len
            for index, sigma in enumerate(args.denoise_sigmas):
                t = torch.full((1, 3), round(sigma * 1000),
                               device='cuda', dtype=torch.long)
                flow = model(z, t=t, context=prompt, seq_len=seq_len,
                             kv_cache=kv, crossattn_cache=cross,
                             current_start=start)
                x0 = z - sigma * flow
                if index + 1 < len(args.denoise_sigmas):
                    noise = torch.randn(x0.shape, device='cuda',
                                        dtype=torch.bfloat16, generator=rollout_rng)
                    z = renoise(x0, noise, args.denoise_sigmas[index + 1])
            chunks.append(x0.cpu().half())
            noise = torch.randn(x0.shape, device='cuda',
                                dtype=torch.bfloat16, generator=context_rng)
            cached = renoise(x0, noise, args.context_sigma)
            model(cached, t=context_t, context=prompt, seq_len=seq_len,
                  kv_cache=kv, crossattn_cache=cross,
                  current_start=start)
    latent = torch.cat(chunks, dim=2)
    metric = transition_rmse(latent, 3)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(latent=latent, adapter=str(Path(args.adapter).resolve()),
                    seed=args.seed, transition=metric),
               destination.with_suffix('.pt'))
    del model, params, kv, cross, chunks, prompt
    gc.collect()
    torch.cuda.empty_cache()

    from wan.modules.vae import WanVAE

    vae = WanVAE(vae_pth=args.vae, z_dim=16, dtype=torch.bfloat16, device='cuda')
    with torch.no_grad():
        decoded = vae.decode([latent[0].to('cuda')])[0]
    decoded = ((decoded.permute(1, 2, 3, 0) + 1.0) * 127.5).clamp(0, 255)
    frames = decoded.byte().cpu().numpy()
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'),
                             24.0, (frames.shape[2], frames.shape[1]))
    if not writer.isOpened():
        raise RuntimeError(f'cannot open output {destination}')
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(json.dumps(dict(output=str(destination), pixel_frames=len(frames),
                          transition=metric), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/data/yg/migrated/home/yg/models/wan2.1-t2v-1.3b')
    parser.add_argument('--ode-checkpoint', default='checkpoints/ode_init.pt')
    parser.add_argument('--prompt-embedding', default='/tmp/wan-liveact-latents/prompt.pt')
    parser.add_argument('--vae', default='/data/yg/migrated/home/yg/models/wan2.1-t2v-1.3b/Wan2.1_VAE.pth')
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--latent-frames', type=int, default=48)
    parser.add_argument('--context-sigma', type=float, default=0.1)
    parser.add_argument('--denoise-sigmas', type=float, nargs='+',
                        default=[1.0, 0.9375, 5.0 / 6.0, 0.625])
    args = parser.parse_args()
    if args.latent_frames < 3 or args.latent_frames % 3:
        raise ValueError('latent frames must be a positive multiple of three')
    generate(args)


if __name__ == '__main__':
    main()
