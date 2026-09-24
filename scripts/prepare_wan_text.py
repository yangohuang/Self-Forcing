"""Cache one real UMT5-XXL text condition before 4090 generator training."""

import argparse
import json
from pathlib import Path

import torch


DEFAULT_T5 = '/data/yg/migrated/home/yg/shared/models/echomimicv3/Wan2.1-Fun-V1.1-1.3B-InP'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', default='A person faces the camera, speaks naturally, and gestures with the hands.')
    parser.add_argument('--t5-dir', default=DEFAULT_T5)
    parser.add_argument('--output', default='/tmp/wan-liveact-latents/prompt.pt')
    args = parser.parse_args()
    from wan.modules.t5 import T5EncoderModel

    root = Path(args.t5_dir)
    # Upstream constructs on `device` before converting dtype. Constructing
    # float32 UMT5 directly on a 24-GiB GPU OOMs during that conversion.
    encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device='cpu',
                             checkpoint_path=str(root / 'models_t5_umt5-xxl-enc-bf16.pth'),
                             tokenizer_path=str(root / 'google/umt5-xxl'))
    encoder.model.to('cuda')
    encoder.device = 'cuda'
    with torch.no_grad():
        embedding = encoder([args.prompt], device='cuda')[0].cpu()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(prompt=args.prompt, embedding=embedding), path)
    print(json.dumps(dict(prompt=args.prompt, shape=list(embedding.shape), path=str(path))))


if __name__ == '__main__':
    main()
