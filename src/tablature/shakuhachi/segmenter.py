import numpy as np
from commonfunctions import histogram, get_line_indices


class ShakuhachiSegmenter:
    """Segment vertical shakuhachi notation into measure boxes."""

    def __init__(self, bin_img):
        self.bin_img = bin_img
        self.segment()

    def segment(self):
        # Histogram along columns to find vertical boundaries
        hist = histogram(self.bin_img.T, 0.8)
        indices = get_line_indices(hist)
        if len(indices) < 2:
            self.regions_with_staff = [self.bin_img]
            self.most_common = 0
            return

        lines = []
        for idx in indices:
            lines.append(((idx, 0), (idx, self.bin_img.shape[0] - 1)))

        end_of_staff = []
        for idx, line in enumerate(lines):
            if idx > 0 and (line[0][0] - end_of_staff[-1][0] < 20):
                continue
            x0, y0 = line[0]
            end_of_staff.append((x0, y0))

        regions = []
        for i in range(len(end_of_staff) - 1):
            x0 = end_of_staff[i][0]
            x1 = end_of_staff[i + 1][0]
            region = self.bin_img[:, x0:x1]
            regions.append(region)
        self.regions_with_staff = regions
        self.most_common = 0

