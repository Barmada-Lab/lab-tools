from collections import defaultdict
from pathlib import Path

from skimage import feature, filters, exposure
import numpy as np
import tifffile

from . import measurements

def get_puncta_blobs(img):
    padded = np.pad(img, 32, mode="edge")
    lowpass = filters.butterworth(padded, cutoff_frequency_ratio=0.1, high_pass=False)
    highpass = filters.butterworth(lowpass, cutoff_frequency_ratio=0.25)
    unpadded = highpass[32:-32,32:-32]
    l, h = np.percentile(unpadded, (10,99.9))
    rescaled = exposure.rescale_intensity(unpadded, in_range=(l,h), out_range=np.uint8)
    blobs_log = feature.blob_log(rescaled, max_sigma=5, threshold=0.5)
    return blobs_log

def label_puncta(img, blobs_log):
    ixs = np.indices(img.shape)
    segmented = np.zeros_like(img)
    for idx, blob in enumerate(blobs_log):
        y, x, r = map(int, blob)
        segmented[y-r:y+r+1,x-r:x+r+1] = idx
    return segmented

def measure_experiment(experiment_path: Path):

    full_cell_measurements = []
    puncta_measurements = []
    paths = list((experiment_path / "processed_imgs" / "segmented").glob("*.tif"))

    for path in paths:
        labeled = tifffile.imread(path)
        rawpath = experiment_path / "raw_imgs"
        channels = ["Cy5", "DAPI", "RFP", "GFP"]
        well = path.name.replace(".tif","")
    
        well_measurements = defaultdict(dict)
        for channel in channels:
            img_path = next((rawpath / channel).glob(f"**/{path.name}"))
            img = tifffile.imread(img_path)
    
            avg = measurements.avg(labeled, img)
            median = measurements.avg(labeled, img)
            std = measurements.avg(labeled, img)
            area = measurements.area(labeled, img)
    
            for id in np.unique(labeled[np.where(labeled !=0)]): # type: ignore
                well_measurements[id][f"{channel}-avg"] = avg[id]
                well_measurements[id][f"{channel}-median"] = median[id]
                well_measurements[id][f"{channel}-std"] = std[id]
                well_measurements[id][f"{channel}-area"] = area[id]
    
        for id, measurement_dict in well_measurements.items():
            row = {
                "well": well,
                "cell_id": id,
                **measurement_dict
            }
            full_cell_measurements.append(row)
    
        cy5 = tifffile.imread(next((rawpath / "Cy5").glob(f"**/{path.name}")))
        puncta_blobs = get_puncta_blobs(cy5)
        puncta_labeled = label_puncta(cy5, puncta_blobs)
        avgs = measurements.avg(puncta_labeled, cy5)
        for puncta_id in np.unique(puncta_labeled[np.where(puncta_labeled != 0)]):
            for cell_id in np.unique(labeled[np.where(labeled !=0)]): # type: ignore
                puncta_mask = puncta_labeled == puncta_id
                cell_mask = labeled == cell_id
                if (puncta_mask * cell_mask).sum() > 0:
                    avg = avgs[puncta_id]
                    _, _, r = puncta_blobs[puncta_id]
                    row = {
                        "well": well,
                        "cell_id": cell_id,
                        "puncta_id": puncta_id,
                        "puncta_radius": r,
                        "Cy5-avg": avg
                    }
                    puncta_measurements.append(row)