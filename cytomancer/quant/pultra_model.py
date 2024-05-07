from skimage.measure import regionprops
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import pandas as pd
import numpy as np


class Pultra_Signal_Area_FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features from image stacks and segmented nuclei expressing the pultra construct.

    Expects a dataframe with the following columns:
    - labels: segmented nuclei
    - gfp: GFP intensity image
    - rfp: RFP intensity image
    - dapi: DAPI intensity image
    """

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return [
            "dapi_signal",
            "gfp_signal",
            "rfp_signal",
            "size"
        ]

    def transform_row(self, row):
        labels = row["labels"]
        dapi = row["dapi"]
        gfp = row["gfp"]
        rfp = row["rfp"]

        dapi_median = np.median(dapi)
        gfp_median = np.median(gfp)
        rfp_median = np.median(rfp)

        for props in regionprops(labels):
            mask = labels == props.label
            yield {
                "dapi_signal": np.median(dapi[mask]) / dapi_median,
                "gfp_signal": np.median(gfp[mask]) / gfp_median,
                "rfp_signal": np.median(rfp[mask]) / rfp_median,
                "size": mask.sum()
            }

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), f"Expected DataFrame, got {type(X)}"
        assert X.columns.isin(["labels", "gfp", "rfp", "dapi"]).all(), \
            f"Missing columns in input DataFrame; expected ['labels', 'gfp', 'rfp', 'dapi'], got {X.columns.values}"
        vecs = [vec for df in X.apply(self.transform_row, axis=1) for vec in df]  # type: ignore
        return pd.DataFrame.from_records(vecs)


def build_pipeline():
    return make_pipeline(
        Pultra_Signal_Area_FeatureExtractor(),
        StandardScaler(),
        SVC(C=10, gamma='auto', kernel='rbf'))
