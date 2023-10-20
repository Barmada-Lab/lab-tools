import pathlib as pl
from itertools import product
from skimage import exposure # type: ignore

import tifffile
import xarray as xr

def write_tiffs(output_dir: pl.Path, arr: xr.DataArray, core_dims: list[str]):
    var_dims = list(arr.dims)
    for dim in core_dims:
        var_dims.remove(dim)
    ndi = list(product(*[arr.indexes[dim].values for dim in var_dims]))
    for coords in ndi:
        chunk = arr.sel(dict(zip(var_dims, coords)))
        filename = "-".join(map(str, coords)) + ".tif"
        print(f"Writing {filename} to {output_dir}")
        tifffile.imwrite(output_dir / filename, chunk)