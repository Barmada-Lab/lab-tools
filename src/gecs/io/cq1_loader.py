import pathlib as pl
import re
import warnings

from skimage import exposure # type: ignore
import dask.array as da
import dask
import xarray as xr
import numpy as np
import ome_types
import tifffile

def read_series(img):
    try:
        arr = img.asarray()
        return exposure.rescale_intensity(arr, out_range=np.float32)
    except (ValueError, NameError, FileNotFoundError) as e:
        warnings.warn(f"Error reading {img.name}: {e}\nThis field will be filled based on surrounding fields and timepoints.")
        arr = np.zeros(img.shape, dtype=np.float32)
        arr[:] = np.nan
        return arr

def read_cq1_experiment(base_path: pl.Path):

    ome_tiff = base_path / "MeasurementResults.ome.tif"
    with tifffile.TiffFile(ome_tiff) as tiff:

        if tiff.ome_metadata is None:
            raise ValueError(f"Missing OME metadata in {ome_tiff}")

        acquisition_meta = ome_types.from_xml(tiff.ome_metadata)

        # Given a string like W14(C2R2),A1,F1, picks out the region index (14) and field idx
        regex = re.compile(
            r"W(?P<region_idx>\d+)\,A(?P<unknown>\d+)\,F(?P<field_idx>\d+)")

        for series in tiff.series[1:]: # skip the first image, which is metadata

            if series.name is None: 
                raise ValueError(f"Img is missing name: {series}")

            search = regex.search(series.name)

            if search is None: 
                raise ValueError(f"Couldn't find region index in {series.name}")

            region_idx = int(search.group("region_idx")) - 1 # 1-indexed in the name, 0-indexed in the array
            field_idx = int(search.group("field_idx")) - 1

            try:
                img_meta = next(img for img in acquisition_meta.images if img.name == series.name)
            except StopIteration:
                raise ValueError(f"Missing metadata for {series.name}")
            
            channels = [channel.name for channel in img_meta.pixels.channels]
            axes = [ax for ax in series.axes]

            arr = xr.DataArray(
                da.from_delayed(
                    dask.delayed(read_series)(series), series.shape, dtype=np.float32),
                dims=axes,
                coords={
                    "channel": channels
                }
            )
