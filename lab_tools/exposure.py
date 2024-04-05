from skimage import exposure  # type: ignore
import xarray as xr

from .util import apply_ufunc_xy
from .experiment import Axes


def equalize_adapthist(arr: xr.DataArray, kernel_size: int | None = None, clip_limit=0.01, nbins=256):

    if kernel_size is None:
        kernel_size = arr.sizes[Axes.X] // 20

    return apply_ufunc_xy(
        exposure.equalize_adapthist,
        arr,
        ufunc_kwargs=dict(kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins))
