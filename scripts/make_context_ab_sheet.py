"""Make a compact, labeled visual comparison from two paired MP4 outputs."""

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def selected_frames(path: Path, indices: list[int], width: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'cannot open {path}')
    result = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f'cannot decode frame {index} in {path}')
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            height = round(image.height * width / image.width)
            result.append(image.resize((width, height)))
    finally:
        capture.release()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--self', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--indices', type=int, nargs='+',
                        default=[0, 24, 48, 72, 96, 120, 144, 168])
    args = parser.parse_args()
    width = 210
    rows = [("Ground-truth-history training", selected_frames(Path(args.teacher), args.indices, width)),
            ("Generated-history training", selected_frames(Path(args.self), args.indices, width))]
    image_height = rows[0][1][0].height
    margin = 38
    sheet = Image.new('RGB', (width * len(args.indices),
                              (image_height + margin) * len(rows)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row_index, (label, images) in enumerate(rows):
        y = row_index * (image_height + margin)
        draw.text((8, y + 4), label, fill='black')
        for col, (index, image) in enumerate(zip(args.indices, images)):
            sheet.paste(image, (col * width, y + margin))
            draw.text((col * width + 8, y + 21), f'{index / 24:.1f}s', fill='black')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(output)


if __name__ == '__main__':
    main()
