from typing import Iterable, IO

from commonfunctions import gray_img
from pre_processing import (
    IsHorizontal,
    deskew,
    rotation,
    get_gray,
    get_thresholded,
    get_closer,
)
from fit import predict
import numpy as np
from skimage.filters import threshold_otsu
from glob import glob
from skimage import io
import argparse

from .config import ShakuhachiConfig

from .segmenter import ShakuhachiSegmenter

# Simple mapping from shakuhachi symbols to MIDI note names
SHAKU_MAP: dict[str, str] = {
    'ro': 'C4/4',
    'tsu': 'D4/4',
    're': 'E4/4',
    'chi': 'F4/4',
}


def recognize_stub(out_file: IO[str], regions: Iterable[np.ndarray], cfg: ShakuhachiConfig) -> None:
    """Recognize regions using the existing classifier.

    Parameters
    ----------
    out_file:
        Open file handle for writing recognized labels.
    regions:
        Iterable of binary numpy arrays corresponding to notation boxes.
    cfg:
        Runtime configuration.
    """

    for region in regions:
        labels = predict(
            (255 * (1 - region)).astype(np.uint8),
            model_path=cfg.model_path,
            dataset_path=cfg.dataset_path,
        )
        label = labels[0]
        out_file.write(SHAKU_MAP.get(label, f"{label}"))
        out_file.write("\n")


def main(input_path: str, output_path: str, cfg: ShakuhachiConfig) -> None:
    """Run the Shakuhachi OMR pipeline."""

    imgs_path = sorted(glob(f"{input_path}/*"))
    for img_path in imgs_path:
        img_name = img_path.split('/')[-1].split('.')[0]
        out_file = open(f"{output_path}/{img_name}.txt", "w")
        img = io.imread(img_path)
        img = gray_img(img)
        horizontal = IsHorizontal(img)
        if not horizontal:
            theta = deskew(img)
            img = rotation(img, theta)
            img = get_gray(img)
            img = get_thresholded(img, threshold_otsu(img))
            img = get_closer(img)

        bin_img = get_thresholded(get_gray(img), threshold_otsu(get_gray(img)))

        segmenter = ShakuhachiSegmenter(bin_img)
        recognize_stub(out_file, segmenter.regions_with_staff, cfg)
        out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfolder", help="Input folder")
    parser.add_argument("outputfolder", help="Output folder")
    parser.add_argument(
        "--model-path",
        default="trained_models/shakuhachi_model.sav",
        help="Trained model file",
    )
    parser.add_argument(
        "--dataset-path",
        default="train_data/shakuhachi",
        help="Training data directory",
    )
    args = parser.parse_args()
    cfg = ShakuhachiConfig(model_path=args.model_path, dataset_path=args.dataset_path)
    main(args.inputfolder, args.outputfolder, cfg)
