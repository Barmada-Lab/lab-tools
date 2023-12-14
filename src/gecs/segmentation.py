from itertools import product
import torch
import pathlib as pl
from skimage import filters, morphology, exposure, segmentation # type: ignore
from cellpose.models import CellposeModel
import dask
from dask.distributed import Lock
from dask.graph_manipulation import bind
import dask.array as da
import xarray as xr
import numpy as np

from .experiment import Axes

def segment_logmaxed_stack(
        arr: xr.DataArray, 
        min_dia: int = 20):
    def _segment_logmaxed_stack(logmaxed):
        thresh = filters.threshold_otsu(logmaxed)
        prelim = logmaxed > thresh
        se = morphology.disk(min_dia // 2)
        opened = np.array(
            [morphology.binary_opening(frame, se) for frame in prelim])
        return opened
    return xr.apply_ufunc(
        _segment_logmaxed_stack,
        arr,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        output_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        dask_gufunc_kwargs=dict(),
        output_dtypes=[bool],
        dask="parallelized",
        vectorize=True)

def annotate_segmentation(
        raw: xr.DataArray, 
        segmented: xr.DataArray, 
        color=(1,1,0)):
    def _annotate_segmentation(raw, segmented):
        rescaled = exposure.rescale_intensity(raw, out_range='uint8')
        marked = segmentation.mark_boundaries(
            rescaled, segmented, color=color, mode="thick")
        marked = exposure.rescale_intensity(marked, out_range='uint8')
        return marked
    return xr.apply_ufunc(
        _annotate_segmentation,
        raw,
        segmented,
        input_core_dims=[[Axes.Y, Axes.X],[Axes.Y, Axes.X]],
        output_core_dims=[[Axes.Y, Axes.X, Axes.RGB]],
        dask_gufunc_kwargs=dict(output_sizes={Axes.RGB: 3}),
        output_dtypes=[np.uint8],
        dask="parallelized",
        vectorize=True)

def annotate_labelled_segmentation(
        raw: xr.DataArray, 
        segmented: xr.DataArray, 
        colors: dict[int, tuple[int,int,int]] = {},
        default_color=(1,1,0)):
    def _annotate_segmentation(raw, labeled):
        if labeled.dtype == bool:
            labeled = labeled.astype(np.uint8)
        marked = raw.copy()
        for id in np.unique(labeled):
            if id == 0:
                continue
            color = colors.get(id, default_color)
            marked = segmentation.mark_boundaries(
                marked, labeled == id, color=color, mode="thick")
        rescaled = exposure.rescale_intensity(marked, out_range='uint8')
        return rescaled
    return xr.apply_ufunc(
        _annotate_segmentation,
        raw,
        segmented,
        input_core_dims=[[Axes.Y, Axes.X],[Axes.Y, Axes.X]],
        output_core_dims=[[Axes.Y, Axes.X, Axes.RGB]],
        dask_gufunc_kwargs=dict(
            output_sizes={Axes.RGB: 3}, 
            ),
        output_dtypes=[np.uint8],
        dask="parallelized",
        vectorize=True)

def label(segmented: xr.DataArray):
    def _label(segmented):
        return morphology.label(segmented)
    return xr.apply_ufunc(
        _label,
        segmented,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        output_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        output_dtypes=[int],
        dask="parallelized",
        vectorize=True)

def segment_clahed_imgs(arr: xr.DataArray, model_path: str):
    """ This will likely work only on a single node cluster """
    """ It is an absolute bastard of a hack """
    """ But it works """
    
    model = CellposeModel(gpu=True, pretrained_model=model_path) # type: ignore

    def _segment_clahe(batch):
        batch = [frame for frame in batch]
        model_lock = Lock("segment_clahe")
        model_lock.acquire()
        preds = model.eval(
            batch, 
            batch_size=64,
            channels=[0, 0],
            normalize=False)[0]
        model_lock.release()
        return np.array(preds)

    return xr.apply_ufunc(
        _segment_clahe,
        arr,
        input_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        output_core_dims=[[Axes.TIME, Axes.Y, Axes.X]],
        output_dtypes=[int],
        dask="parallelized",
        vectorize=True)