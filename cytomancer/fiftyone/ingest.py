import logging
import uuid
from pathlib import Path
from more_itertools import chunked

from tqdm import tqdm
import tifffile
import fiftyone as fo
from PIL import Image
from distributed import as_completed, get_client
from skimage import exposure  # type: ignore

from cytomancer.config import config
from cytomancer.io import cq1_loader

logger = logging.getLogger(__name__)


def ingest_cq1_longitudinal(experiment_path: Path):

    client = get_client()

    experiment_name = experiment_path.name
    cache_dir = config.fo_cache / experiment_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    if fo.dataset_exists(experiment_name):
        raise ValueError("Dataset already exists; delete before re-ingesting")

    dataset = fo.Dataset(name=experiment_name)

    df, _, _ = cq1_loader.get_experiment_df(experiment_path)
    axes = df.index.names

    def create_sample(coord_row):
        coords, row = coord_row
        path = row["path"]
        tags_dict = dict(zip(axes, coords))

        arr = tifffile.imread(path)
        rescaled = exposure.rescale_intensity(arr, out_range="uint8")
        image = Image.fromarray(rescaled)
        png_path = cache_dir / f"{uuid.uuid4()}.png"
        image.save(png_path, format="PNG")

        return (png_path, tags_dict)

    chunks = list(chunked(df.iterrows(), 100))
    for chunk in tqdm(chunks):
        for _, result in as_completed(client.map(create_sample, chunk), with_results=True):
            path, tags = result  # type: ignore
            sample = fo.Sample(filepath=path)
            for key, value in tags.items():
                sample[key.name] = value
            dataset.add_sample(sample)
