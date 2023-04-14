import re
import pathlib
from pathlib import Path
from datetime import timedelta
from typing import Callable

from improc.experiment import *
from .mfile import read_mfile
from .util import get_layout_indexing

@dataclass
class LegacyDirDataset(Dataset):
    path: pathlib.Path
    tag_reader: Callable[[Path], list[Tag]]

    @property
    def images(self) -> list[Image]:
        _images = []
        for path in self.path.glob("**/*.tif"):
            tags = self.tag_reader(path)
            _images.append(ImageFile(path, tags))
        return _images

    def write_image(self, image: np.ndarray, tags: list[Tag]) -> Image:
        return NotImplemented("Legacy datasets are readonly :-)")

class LegacyLoader:

    def __init__(self, path: pathlib.Path, scratch_dir: pathlib.Path) -> None:
        super().__init__()
        self.path = path
        self.scratch_dir = scratch_dir
        self.name = path.name
        self.mfile = self._try_load_mfile()

    def _try_load_mfile(self) -> MFSpec:
        experiment_name = self.name
        glob = f"*{experiment_name}.csv"
        try:
            csv = next(self.path.glob(glob))
        except StopIteration:
            raise Exception(f'No csv of the form "{glob}" found in {self.path}')
        mfile = read_mfile(csv)
        if mfile is None:
            raise Exception("ERROR ERROR ERROR ERROR ha ha this is all the information you get.")
        return mfile


    def extract_meta_rawpath(self, path: pathlib.Path) -> list[Tag]:
        """
        Extract image metadata from a standard experiment path
        e.g: experiment_root/raw_imgs/RFP/T1/col_01/A01_01.tif
        """

        pattern = r"""# Verbose regex
            (?P<channel>(GFP)|(RFP)|(Cy5)|(white_light)|(DAPI))/ # Pick out the channel
            T(?P<timepoint>\d+)/                                 # Pick out the time
            col_(?P<col>\d+)/                                    # pick out the row
            (?P<row>[a-z])\d+_(?P<mosaic_idx>\d{2}).tif          # Pick out the row and montage index from the file name
        """

        search = re.search(pattern, path.__str__(), re.IGNORECASE | re.VERBOSE)
        if search is None:
            raise Exception(f"Failed to parse path {path}")

        mosaic_idx = int(search.group("mosaic_idx")) - 1
        order = get_layout_indexing(self.mfile.microscope, self.mfile.montage_dim)
        mosaic_idx_tup = tuple(np.argwhere(order == mosaic_idx)[0])
        mosaic_tag = Mosaic(index = mosaic_idx_tup, overlap = self.mfile.montage_overlap)

        timepoint = int(search.group("timepoint")) - 1
        if 0  <= timepoint < len(self.mfile.imaging_times):
            realtime = self.mfile.imaging_times[timepoint]
        timepoint_tag = Timepoint(timepoint, realtime)

        row = search.group("row")
        col = search.group("col")
        label = row + col
        exposures = next(filter(lambda x: x.label == label, self.mfile.wells)).exposures
        channel_str = search.group("channel")
        exposure_ms = next(filter(lambda x: x.channel == channel_str, exposures)).exposure_ms
        if channel_str == "white_light":
            channel_str = "brightfield"
        exposure_tag = Exposure(duration = timedelta(milliseconds=exposure_ms), channel=Channel(channel_str))

        vertex_tag = Vertex(label)
        planar_tag = Geometry(objective = Objective(self.mfile.objective))

        return [vertex_tag, mosaic_tag, timepoint_tag, exposure_tag, planar_tag]

    def load(self) -> Experiment:
        raw_path = self.path / "raw_imgs"
        raw_dataset = LegacyDirDataset("raw_imgs", raw_path, self.extract_meta_rawpath)
        scratch_datasets = {}
        for path in self.scratch_dir.glob("*"):
            if path.is_dir():
                label = path.name
                scratch_datasets[label] = DirDataset(label, path)
        return Experiment(
            label = self.name,
            datasets = { "raw_imgs": raw_dataset, **scratch_datasets },
            experiment_dir = self.path,
            scratch_dir = self.scratch_dir,
            mfspec = self.mfile
        )
