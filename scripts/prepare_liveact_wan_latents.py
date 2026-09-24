"""Encode fixed 81-frame LiveAct video windows with the official Wan 2.1 VAE."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def read_window(video_path: Path, start: int, frames: int, width: int, height: int):
    if start < 0 or frames < 1:
        raise ValueError('start must be nonnegative and frames positive')
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'cannot open {video_path}')
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    images = []
    try:
        for _ in range(frames):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f'{video_path} has fewer than {start + frames} decodable frames')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images.append(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    return np.stack(images)


def encode_window(args):
    from wan.modules.vae import WanVAE

    source = Path(args.video)
    pixels = read_window(source, args.start, args.frames, args.width, args.height)
    tensor = torch.from_numpy(pixels).permute(3, 0, 1, 2).contiguous().float()
    tensor = (tensor / 127.5 - 1.0).to('cuda')
    vae = WanVAE(vae_pth=args.vae, z_dim=16, dtype=torch.bfloat16, device='cuda')
    with torch.no_grad():
        latent = vae.encode([tensor])[0].cpu().half()
    result = dict(latent=latent,
                  metadata=dict(source=str(source.resolve()),
                                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                start=args.start, frames=args.frames,
                                pixel_hw=[args.height, args.width],
                                latent_shape=list(latent.shape)))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, destination)
    print(json.dumps(result['metadata'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vae', default='/data/yg/migrated/home/yg/models/wan2.1-t2v-1.3b/Wan2.1_VAE.pth')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--frames', type=int, default=81)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--height', type=int, default=832)
    encode_window(parser.parse_args())


if __name__ == '__main__':
    main()
