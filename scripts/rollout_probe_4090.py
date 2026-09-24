"""Bounded single-GPU generated-history backward probe, not DMD training."""

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_MODEL = Path('/data/yg/migrated/home/yg/models/wan2.1-t2v-1.3b')


def renoise(clean: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    """Detach a generated x0 before constructing a context-noise sample."""
    if not 0.0 <= sigma <= 1.0:
        raise ValueError('sigma must be in [0, 1]')
    return (1.0 - sigma) * clean.detach() + sigma * noise


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4):
        super().__init__()
        if rank < 1:
            raise ValueError('rank must be positive')
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 1.0 / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        update = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return base + update.to(base.dtype)


def make_kv_cache(layers: int, batch: int, tokens: int, heads: int, head_dim: int,
                  dtype: torch.dtype, device: str):
    return [dict(k=torch.zeros(batch, tokens, heads, head_dim, dtype=dtype, device=device),
                 v=torch.zeros(batch, tokens, heads, head_dim, dtype=dtype, device=device),
                 global_end_index=torch.zeros(1, dtype=torch.long, device=device),
                 local_end_index=torch.zeros(1, dtype=torch.long, device=device))
            for _ in range(layers)]


def install_lora(model: nn.Module, rank: int) -> list[nn.Parameter]:
    model.requires_grad_(False)
    params = []
    for block in model.blocks:
        for name in ('q', 'v'):
            layer = LoRALinear(getattr(block.self_attn, name), rank)
            setattr(block.self_attn, name, layer)
            params.extend((layer.a, layer.b))
    return params


def checkpoint_metadata(path: Path) -> dict:
    from safetensors import safe_open

    config_file = path / 'config.json'
    weight_file = path / 'diffusion_pytorch_model.safetensors'
    if not config_file.is_file() or not weight_file.is_file():
        raise FileNotFoundError(f'Wan model config or weights absent in {path}')
    config = json.loads(config_file.read_text())
    with safe_open(str(weight_file), framework='pt', device='cpu') as state:
        keys = list(state.keys())
        shape = state.get_slice('blocks.0.self_attn.q.weight').get_shape()
    return dict(path=str(path), bytes=weight_file.stat().st_size,
                tensor_count=len(keys), q_weight_shape=shape, config=config)


def run_probe(args: argparse.Namespace) -> dict:
    model_path = Path(args.model)
    meta = checkpoint_metadata(model_path)
    result = dict(model=meta, seed=args.seed, latent_h=args.latent_h,
                  latent_w=args.latent_w, frames_per_block=args.frames_per_block,
                  history_blocks=args.history_blocks,
                  denoise_sigma=args.denoise_sigma, context_sigma=args.context_sigma,
                  rank=args.rank, dry_run=args.dry_run,
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  torch_version=torch.__version__)
    if args.dry_run:
        return result
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run the probe on the RTX 4090 host')
    from wan.modules.causal_model import CausalWanModel

    device = 'cuda'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    result['gpu'] = torch.cuda.get_device_name(0)
    load_start = time.perf_counter()
    model = CausalWanModel.from_pretrained(str(model_path), torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True).to(device).eval()
    params = install_lora(model, args.rank)
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    torch.cuda.synchronize()
    result['load_seconds'] = time.perf_counter() - load_start
    result['parameters_total'] = sum(p.numel() for p in model.parameters())
    result['parameters_trainable'] = sum(p.numel() for p in params)
    result['loaded_gib'] = torch.cuda.memory_allocated() / 2**30

    batch = 1
    frames = args.frames_per_block
    if args.history_blocks < 1:
        raise ValueError('history_blocks must be positive')
    h, w = args.latent_h, args.latent_w
    tokens_per_frame = h * w // 4
    tokens_per_block = frames * tokens_per_frame
    cache = make_kv_cache(len(model.blocks), batch,
                          (args.history_blocks + 1) * tokens_per_block,
                          model.num_heads, model.dim // model.num_heads,
                          torch.bfloat16, device)
    cross_cache = [dict(is_init=False) for _ in model.blocks]
    context = [torch.randn(32, model.text_dim, device=device, dtype=torch.bfloat16)]
    seq_len = tokens_per_block
    noise2 = torch.randn(batch, model.in_dim, frames, h, w,
                         device=device, dtype=torch.bfloat16)
    target = torch.randn_like(noise2)
    t = torch.full((batch, frames), int(args.denoise_sigma * 1000), device=device, dtype=torch.long)
    context_t = torch.full_like(t, int(args.context_sigma * 1000))

    start = time.perf_counter()
    with torch.no_grad():
        for block_index in range(args.history_blocks):
            noise1 = torch.randn(batch, model.in_dim, frames, h, w,
                                 device=device, dtype=torch.bfloat16)
            block_start = block_index * tokens_per_block
            flow1 = model(noise1, t=t, context=context, seq_len=seq_len,
                          kv_cache=cache, crossattn_cache=cross_cache,
                          current_start=block_start)
            x0_1 = noise1 - args.denoise_sigma * flow1
            history = renoise(x0_1, torch.randn_like(x0_1), args.context_sigma)
            model(history, t=context_t, context=context, seq_len=seq_len,
                  kv_cache=cache, crossattn_cache=cross_cache,
                  current_start=block_start)
    torch.cuda.synchronize()
    result['history_seconds'] = time.perf_counter() - start
    result['history_detached'] = not history.requires_grad
    result['history_cache_tokens'] = cache[0]['global_end_index'].item()

    start = time.perf_counter()
    flow2 = model(noise2, t=t, context=context, seq_len=seq_len,
                  kv_cache=cache, crossattn_cache=cross_cache,
                  current_start=args.history_blocks * tokens_per_block)
    x0_2 = noise2 - args.denoise_sigma * flow2
    loss = F.mse_loss(x0_2.float(), target.float())
    loss.backward()
    torch.cuda.synchronize()
    result['backward_seconds'] = time.perf_counter() - start
    grad_norm = torch.sqrt(sum(p.grad.float().square().sum() for p in params if p.grad is not None))
    result['loss'] = loss.item()
    result['grad_norm'] = grad_norm.item()
    result['finite_loss_and_grad'] = math.isfinite(result['loss']) and math.isfinite(result['grad_norm'])
    result['nonzero_gradient'] = result['grad_norm'] > 0
    optimizer.step()
    torch.cuda.synchronize()
    result['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    result['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
    result['success'] = result['history_detached'] and result['finite_loss_and_grad'] and result['nonzero_gradient']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--latent-h', type=int, default=16)
    parser.add_argument('--latent-w', type=int, default=24)
    parser.add_argument('--frames-per-block', type=int, default=1)
    parser.add_argument('--history-blocks', type=int, default=1)
    parser.add_argument('--denoise-sigma', type=float, default=0.75)
    parser.add_argument('--context-sigma', type=float, default=0.1)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = run_probe(args)
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    print(encoded, flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(encoded + '\n')


if __name__ == '__main__':
    main()
