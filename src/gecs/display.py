from PIL import ImageColor
import cv2
from skimage import exposure, color, util # type: ignore
from pybasic.shading_correction import BaSiC
import pandas as pd
import xarray as xr
import numpy as np

from gecs.experiment import Axes

def rescale_intensity(arr: xr.DataArray, dims: list[Axes], **kwargs):
    def _rescale_intensity(frame, in_percentile: tuple[int,int] | None = None, **kwargs):
        if in_percentile is not None:
            l, h = np.percentile(frame, in_percentile)
            kwargs.pop("in_range", None)
            return exposure.rescale_intensity(frame, in_range=(l, h), **kwargs)
        else:
            return exposure.rescale_intensity(frame, **kwargs)
    return xr.apply_ufunc(
        _rescale_intensity,
        arr,
        kwargs=kwargs,
        input_core_dims=[dims],
        output_core_dims=[dims],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        vectorize=True,
        dask="parallelized")

def clahe(arr: xr.DataArray):
    def _clahe(frame):
        rescaled = exposure.rescale_intensity(frame, out_range=np.uint8)
        blurred = cv2.medianBlur(rescaled, 7)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(64,64))
        return clahe.apply(blurred)
    return xr.apply_ufunc(
        _clahe,
        arr,
        input_core_dims=[[Axes.Y, Axes.X]],
        output_core_dims=[[Axes.Y, Axes.X]],
        vectorize=True,
        dask="parallelized")

def _get_float_color(hexcode: str):
    rgb = tuple(map(float, ImageColor.getcolor(hexcode, "RGB")))
    max_val = max(rgb)
    rgb_corrected = tuple(map(lambda x: x / max_val, rgb))
    return rgb_corrected

def apply_psuedocolor(arr: xr.DataArray):

    if "metadata" in arr.attrs:
        channels = arr.attrs["metadata"]["metadata"].channels
        color_codes = {}
        for channel in channels:
            intcode = channel.channel.colorRGB
            rgb = (intcode & 255, (intcode >> 8) & 255, (intcode >> 16) & 255)
            max_val = max(rgb)
            float_color = tuple(map(lambda x: x / max_val, rgb))
            color_codes[channel.channel.name] = float_color
    else:
        color_codes = {
            "DAPI": _get_float_color("#007fff"),
            "RFP": _get_float_color("#ffe600"),
            "GFP": _get_float_color("#00ff00"),
            "Cy5": _get_float_color("#ff0000"),
            "white_light": _get_float_color("#ffffff"),
        }

    def _rgb(frame, channel):
        color_code = color_codes.get(str(channel), (1.0,1.0,1.0))
        float_frame = util.img_as_float(frame)
        rgb_frame = color.gray2rgb(float_frame)
        colored = rgb_frame * color_code
        return exposure.rescale_intensity(colored, out_range="uint8")

    rgb = xr.apply_ufunc(
        _rgb, 
        arr,
        arr[Axes.CHANNEL],
        input_core_dims=[[Axes.Y, Axes.X], []], 
        output_core_dims=[[Axes.Y, Axes.X, Axes.RGB]],
        dask_gufunc_kwargs=dict(output_sizes={Axes.RGB: 3}),
        output_dtypes=[np.uint8],
        vectorize=True, 
        dask="parallelized")
    
    return rgb.transpose(..., Axes.RGB)

def illumination_correction(arr: xr.DataArray, dims: list[str]):
    assert Axes.X in dims and Axes.Y in dims, "x and y dimensions must be specified"
    assert len(dims) == 3, "Must provide a third dimension to iterate over"
    def _illumination_correction(stack):
        basic = BaSiC(stack)
        basic.prepare()
        basic.run()
        return np.array([basic.normalize(frame) for frame in stack])

    return xr.apply_ufunc(
        _illumination_correction,
        arr,
        input_core_dims=[dims],
        output_core_dims=[dims],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        vectorize=True,
        dask="parallelized")

def stitch(arr: xr.DataArray, trim: float = 0.05):
    # TODO: arrange tiles correctly
    trimmed = arr.sel({
        Axes.Y: slice(int(arr[Axes.Y].size * trim), int(arr[Axes.Y].size * (1 - trim))), 
        Axes.X: slice(int(arr[Axes.X].size * trim), int(arr[Axes.X].size * (1 - trim)))
    })
    field_dim = np.sqrt(arr[Axes.FIELD].size).astype(int)
    mi = pd.MultiIndex.from_product(
        (range(field_dim), range(field_dim)), names=["fx", "fy"])
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mi,Axes.FIELD)
    trimmed = trimmed.assign_coords(mindex_coords).unstack(Axes.FIELD)

    x_stitched = xr.concat(trimmed.transpose("fx", ..., Axes.Y, Axes.X)[::-1], dim=Axes.Y)
    stitched = xr.concat(x_stitched.transpose("fy", ..., Axes.X, Axes.Y)[::-1], dim=Axes.X)
    return stitched.drop(["fx", "fy"]).chunk({
        Axes.TIME: -1,
        Axes.Y: -1,
        Axes.X: -1,
    })