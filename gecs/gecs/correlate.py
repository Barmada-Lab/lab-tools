from pathlib import Path

from PIL import Image
import pandas as pd
import numpy as np

def correlate_rois(roi_img1, roi_img2) -> pd.DataFrame:
    correlates = []
    intersection = roi_img1 * roi_img2
    for id in np.unique(intersection[np.where(intersection != 0)]):
        img1_id = roi_img1[np.where(intersection == id)][0]
        img2_id = roi_img2[np.where(intersection == id)][0]
        correlates.append((img1_id, img2_id))
    return pd.DataFrame(sorted(correlates), columns=['roi1', 'roi2'])

def cli_entry(args):
    rois_1 = { path.name:path for path in args.roi_dir1.glob("*.tif") }
    rois_2 = { path.name:path for path in args.roi_dir2.glob("*.tif") }

    assert rois_1.keys() == rois_2.keys(), "ROI sets do not match"

    df = pd.DataFrame()
    for roi_name in rois_1:
        roi_1_img = Image.open(rois_1[roi_name])
        roi_2_img = Image.open(rois_2[roi_name])

        roi_1 = np.array(roi_1_img)
        roi_2 = np.array(roi_2_img)

        correlations = correlate_rois(roi_1, roi_2)
        correlations.insert(0, column='roi_name', value=roi_name)
        df = pd.concat((df, correlations), axis=0)

    output_path = args.output if args.output is not None else Path.cwd() / 'correlations.csv'
    df.to_csv(output_path, index=False)