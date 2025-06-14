from commonfunctions import gray_img
from pre_processing import IsHorizontal, deskew, rotation, get_gray, get_thresholded, get_closer
from fit import predict
from skimage.filters import threshold_otsu
from glob import glob
from skimage import io
import argparse

from .segmenter import ShakuhachiSegmenter

# Simple mapping from shakuhachi symbols to MIDI note names
SHAKU_MAP = {
    'ro': 'C4/4',
    'tsu': 'D4/4',
    're': 'E4/4',
    'chi': 'F4/4',
}


def recognize_stub(out_file, regions):
    """Placeholder recognition using the existing classifier."""
    for region in regions:
        labels = predict((255 * (1 - region)).astype(np.uint8))
        label = labels[0]
        out_file.write(SHAKU_MAP.get(label, f"{label}"))
        out_file.write("\n")


def main(input_path, output_path):
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
        recognize_stub(out_file, segmenter.regions_with_staff)
        out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfolder", help="Input folder")
    parser.add_argument("outputfolder", help="Output folder")
    args = parser.parse_args()
    main(args.inputfolder, args.outputfolder)
