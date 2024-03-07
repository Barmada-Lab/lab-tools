from skimage import filters, morphology, exposure, segmentation  # type: ignore
import xarray as xr
import numpy as np

from lab_tools.experiment import Axes


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
        color=(1, 1, 0)):

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
        input_core_dims=[[Axes.Y, Axes.X], [Axes.Y, Axes.X]],
        output_core_dims=[[Axes.Y, Axes.X, Axes.RGB]],
        dask_gufunc_kwargs=dict(output_sizes={Axes.RGB: 3}),
        output_dtypes=[np.uint8],
        dask="parallelized",
        vectorize=True)


def annotate_labelled_segmentation(
        raw: xr.DataArray,
        segmented: xr.DataArray,
        colors: dict[int, tuple[int, int, int]] = {},
        default_color=(1, 1, 0)):

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
        input_core_dims=[[Axes.Y, Axes.X], [Axes.Y, Axes.X]],
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
