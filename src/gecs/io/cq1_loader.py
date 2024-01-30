import pathlib as pl
from itertools import product
import xml.etree.ElementTree as xml
import warnings

from gecs.experiment import Axes
from datetime import datetime
from skimage import exposure  # type: ignore
import tifffile
import dask.array as da
import dask
import pandas as pd
import ome_types
import xarray as xr
import numpy as np
import re

CQ1_ACQUISITION_DIR_REGEX = r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})T(?P<hr>\d{2})(?P<min>\d{2})(?P<sec>\d{2})_(?P<plate_name>.*)$"
CQ1_WELLPLATE_NAME_REGEX = r"^W(?P<well_idx>\d*)\(.*\),A.*,F(?P<field_idx>\d*)$"

CHANNEL_EX_EM_LUT = {
    (405, 447): "DAPI",
    (488, 525): "GFP",
    (561, 617): "RFP"
}

PLATE_WELL_LUT = {
    (8, 12): [r+c for r, c in product(["A", "B", "C", "D", "E", "F", "G", "H"], ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])],
}


def read_series(img):
    try:
        arr = img.asarray()
        return exposure.rescale_intensity(arr, out_range=np.float32)
    except (ValueError, NameError, FileNotFoundError) as e:
        warnings.warn(f"Error reading {img.name}: {e}\nThis field will be filled based on surrounding fields and timepoints.")
        arr = np.zeros(img.shape, dtype=np.float32)
        arr[:] = np.nan
        return arr


def _try_parse_dir(path: pl.Path) -> datetime | None:
    if (match := re.match(CQ1_ACQUISITION_DIR_REGEX, path.name)) is not None:
        return datetime(
            year=int(match["year"]),
            month=int(match["month"]),
            day=int(match["day"]),
            hour=int(match["hr"]),
            minute=int(match["min"]),
            second=int(match["sec"]),
        )
    return None


def load_acquisition(path: pl.Path, ome_xml_filename: str | None = None) -> xr.DataArray:  # noqa: C901, get bent flake8
    """
    Load an individual CQ1 acquisition directory. If no argument is provided for
    ome_xml_filename, the function will automatically load an ome.xml file from
    the directory based on the following priority:

        1. MeasurementResult.ome.xml
        2. MeasurementResultMIP.ome.xml
        3. MeasurementResultSUM.ome.xml
    """

    if ome_xml_filename is not None:
        if not (xml_path := path / ome_xml_filename).exists():
            raise ValueError(f"Could not find {ome_xml_filename} in {path}.")
        ome_xml = ome_types.from_xml(xml_path)
    elif (xml_path := path / "MeasurementResult.ome.xml").exists():
        ome_xml_filename = "MeasurementResult.ome.xml"
        ome_xml = ome_types.from_xml(xml_path)
    elif (xml_path := path / "MeasurementResultMIP.ome.xml").exists():
        ome_xml_filename = "MeasurementResultMIP.ome.xml"
        ome_xml = ome_types.from_xml(xml_path)
    elif (xml_path := path / "MeasurementResultSUM.ome.xml").exists():
        ome_xml_filename = "MeasurementResultSUM.ome.xml"
        ome_xml = ome_types.from_xml(xml_path)
    else:
        raise ValueError(f"Could not find an ome.xml file in {path}.")

    result_xml_path = path / "ImagingResult.xml"
    assert result_xml_path.exists(), f"Could not find ImagingResult.xml in {path}."
    result_xml = xml.parse(result_xml_path)

    info = result_xml.find("{http://www.yokogawa.co.jp/LSC/ICMSchema/1.0}ResultInfo")
    start_time_str = info.get("{http://www.yokogawa.co.jp/LSC/ICMSchema/1.0}BeginTime")  # type: ignore
    end_time_str = info.get("{http://www.yokogawa.co.jp/LSC/ICMSchema/1.0}EndTime")  # type: ignore
    fmt = "%Y-%m-%dT%H:%M:%S"
    start_time = datetime.strptime(start_time_str.split(".")[0], fmt)  # type: ignore
    end_time = datetime.strptime(end_time_str.split(".")[0], fmt)  # type: ignore
    acquisition_delta = start_time - end_time

    plate = ome_xml.plates[0]
    rows, cols = plate.rows, plate.columns
    if (rows, cols) in PLATE_WELL_LUT:
        wells = PLATE_WELL_LUT[(rows, cols)]  # type: ignore
    else:
        warnings.warn("Could not find well names for this plate size. Falling back to integer-based well names. Consider adding plate size to PLATE_WELL_LUT in cq1_loader.py.")
        wells = [f"{d:02}_{c:02}" for d, c in product(range(rows), range(cols))]  # type: ignore

    ex_px = ome_xml.images[1].pixels

    channels = []
    for channel in ex_px.channels:
        if channel.illumination_type.value == "Epifluorescence":  # type: ignore
            ex, em = int(channel.excitation_wavelength), int(channel.emission_wavelength)  # type: ignore
            channel_str = CHANNEL_EX_EM_LUT.get((ex, em), f"{ex}nm/{em}nm")  # type: ignore
            channels.append(channel_str)
        else:
            channels.append(channel.contrast_method.value)  # type: ignore

    if len(channels) != len(set(channels)):
        # fallback to integer-based channel names
        channels = list(range(len(channels)))

    shape = (ex_px.size_x, ex_px.size_y)
    attrs = dict(
        ome_xml_filename=ome_xml_filename,
        px_size_x=ex_px.physical_size_x,
        px_size_y=ex_px.physical_size_y,
        px_size_z=ex_px.physical_size_z)  # note: for MIPs, means the z step size of the original images

    records = []
    for image in ome_xml.images[1:]:  # skip the first image, which contains only metadata
        assert image.name is not None, f"Image {image.id} has no name."
        image_match = re.match(CQ1_WELLPLATE_NAME_REGEX, image.name)
        assert image_match is not None, f"Failed to parse image name: {image.name}"

        well_idx = int(image_match["well_idx"]) - 1
        well_label = wells[well_idx]
        field_idx_label = image_match["field_idx"]

        pixels = image.pixels

        for plane, data in zip(pixels.planes, pixels.tiff_data_blocks):
            z = plane.the_z
            t = plane.the_t
            c = channels[plane.the_c]
            assert z is not None and t is not None and c is not None, f"Image plane is missing coordinate information: {plane}."
            assert data.uuid is not None, f"Data block {data.id} has no UUID."
            image_path = data.uuid.file_name
            assert image_path is not None, f"Data block {data.id} has no file name."
            records.append({
                Axes.TIME: t,
                Axes.CHANNEL: c,
                Axes.REGION: well_label,
                Axes.FIELD: field_idx_label,
                Axes.Z: z,
                "path": path / image_path,
            })

    df = pd.DataFrame.from_records(records)
    ts = df[Axes.TIME].unique().size
    acq_delta = acquisition_delta / ts
    df[Axes.TIME] = df[Axes.TIME].map(lambda t: start_time + acq_delta * t).astype("datetime64[ns]")
    mi = pd.MultiIndex.from_frame(df.drop(["path"], axis=1))

    def read_indexed_ims(recurrence):
        """
        Recursively read and stack images from a sorted hierarchical index
        in a breadth-first manner.
        """
        if type(recurrence) is pd.Series:
            path = recurrence["path"]
            return da.from_delayed(dask.delayed(tifffile.imread)(path), shape, dtype=np.uint16)
        else:  # type(recurrence) is pd.DataFrame
            if type(recurrence.index) is pd.MultiIndex:
                level = recurrence.index.levels[0]  # type: ignore
            else:  # type(recurrence.index) is pd.Index
                level = recurrence.index.values
            return da.stack([read_indexed_ims(recurrence.loc[idx]) for idx in level])

    df = pd.DataFrame(
        index=mi,
        data=df[["path"]].values,
        columns=["path"]
    ).sort_index()
    arr = read_indexed_ims(df)

    labels = [Axes.TIME, Axes.CHANNEL, Axes.REGION, Axes.FIELD, Axes.Z]
    arr = xr.DataArray(
        arr,
        dims=labels + [Axes.Y, Axes.X],
        coords=dict((label, val) for label, val in zip(labels, df.index.levels)),  # type: ignore
        attrs=attrs)

    # The above method will produce coordinates of dtype object, which
    # causes issues downstream as it's inconsistent with other experiment loaders. '
    # So we explicitly cast them to strings here /shrug
    arr.coords[Axes.CHANNEL] = arr.coords[Axes.CHANNEL].astype(str)
    arr.coords[Axes.REGION] = arr.coords[Axes.REGION].astype(str)
    arr.coords[Axes.FIELD] = arr.coords[Axes.FIELD].astype(str)
    return arr


def _load_and_tag_acquisition(dt: datetime, path: pl.Path) -> xr.DataArray:
    arr = load_acquisition(path)
    # do stuff with dt
    return arr


def load_cq1(base_path: pl.Path) -> xr.Dataset:
    """Load a CQ1 experiment from a directory.

    Parameters
    ----------
    base_path : pl.Path
        Path to the directory containing the experiment. This can either be a directory
        containing one or more acquisition subdirectories, or an individual acquisition
        directory.

    Returns
    -------
    xr.Dataset
        The experiment data.
    """

    if (dt := _try_parse_dir(base_path)) is not None:
        arr = _load_and_tag_acquisition(dt, base_path)
        return xr.Dataset(dict(intensity=arr))
    else:
        dt_paths = [(_try_parse_dir(d), d) for d in base_path.iterdir() if d.is_dir()]
        dt_paths = [(dt, path) for dt, path in dt_paths if dt is not None]
        dt_paths = sorted(dt_paths, key=lambda d: d[0])
        if any(dt_paths):
            arrs = [_load_and_tag_acquisition(dt, path) for dt, path in dt_paths]
            arr = xr.concat(arrs, dim=Axes.TIME)
            return xr.Dataset(dict(intensity=arr))
        else:
            raise ValueError(f"Could not find any acquisition directories in {base_path}.")
