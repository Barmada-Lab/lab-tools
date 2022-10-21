from pathlib import Path
from read_roi import read_roi_zip
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from collections import defaultdict
from shapely.geometry import Polygon

import numpy as np
import tifffile
import csv
import cv2

import napari


def calc_frac(time, well_id, imgs, cell_rois, nuc_rois):
    cell_nuc_pairs = defaultdict(list)
    for key, cell_roi in cell_rois.items():
        if "x" not in cell_roi:
            continue
        cell_poly = Polygon(list(zip(cell_roi["x"], cell_roi["y"]))).convex_hull
        for nuc_roi in nuc_rois.values():
            if "x" not in nuc_roi:
                continue
            nuc_poly = Polygon(list(zip(nuc_roi["x"], nuc_roi["y"]))).convex_hull
            if cell_poly.intersects(nuc_poly):
                cell_nuc_pairs[key].append(nuc_roi)

    for channel, img in imgs.items():
        for cell_roi_idx, paired_nuc_rois in cell_nuc_pairs.items():
            cell_roi = cell_rois[cell_roi_idx]
            cell_roi = np.array(list(zip(cell_roi["x"], cell_roi["y"])))

            cell_mask = np.zeros_like(img)
            cv2.fillConvexPoly(cell_mask, cell_roi.astype(int), 1)

            nuc_mask = np.zeros_like(img)
            for raw_roi in paired_nuc_rois:
                roi = np.array(list(zip(raw_roi["x"], raw_roi["y"])))
                cv2.fillConvexPoly(nuc_mask, roi.astype(int), 1)

            nuc_filtered = cell_mask & nuc_mask
            cyto = (cell_mask - nuc_filtered)

            nuc_signal = (img * nuc_filtered).sum()
            nuc_area = np.count_nonzero(nuc_filtered)
            cyto_signal = (img * cyto).sum()
            cyto_area = np.count_nonzero(cyto)

            yield {
                "time": time,
                "channel": channel,
                "tile": well_id,
                "id": cell_roi_idx,
                "nuc_signal": nuc_signal,
                "nuc_area": nuc_area,
                "nuc_mean": nuc_signal / nuc_area,
                "cyto_signal": cyto_signal,
                "cyto_area": cyto_area,
                "cyto_mean": cyto_signal / cyto_area,
                "nuc_cyto_mean_ratio": (nuc_signal / nuc_area) / (cyto_signal / cyto_area)
            }

def nuc_frac(experiment_path: Path):
    manualrois = experiment_path / "ManualROIs"
    nucrois = experiment_path / "nucROIs"
    header = ["time", "tile", "id", "channel", "nuc_signal", "nuc_area", "nuc_mean", "cyto_signal", "cyto_mean", "cyto_area", "nuc_cyto_mean_ratio"]
    rows = []
    cell_base = experiment_path / "raw_imgs" / "RFP"
    for t in cell_base.glob("*"):
        for img_path in t.glob("**/*.tif"):
            zip_name = img_path.name.replace(".tif", ".zip")
            rel_path = img_path.relative_to(cell_base).parent
            cell_roi_path = manualrois / rel_path / zip_name
            nuc_roi_path = nucrois / rel_path / zip_name
            if (cell_roi_path.exists() and nuc_roi_path.exists()):
                channels = list((experiment_path / "raw_imgs").glob("*"))
                corresponding_imgs = { channel.name: tifffile.imread(channel / rel_path / img_path.name) for channel in channels }
                cell_roi = read_roi_zip(cell_roi_path)
                nuc_roi = read_roi_zip(nuc_roi_path)
                well_id = img_path.name.strip(".tif")
                rows += list(calc_frac(t.name[1:], well_id, corresponding_imgs, cell_roi, nuc_roi))

    with open(experiment_path / "results" / "nuc_frac.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
