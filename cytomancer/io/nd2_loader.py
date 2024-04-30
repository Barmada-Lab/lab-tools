import pathlib as pl

import xarray as xr
import nd2

from cytomancer.experiment import Axes


def load_nd2(path: pl.Path) -> xr.Dataset:

    arr = nd2.imread(path, dask=True, xarray=True)
    nd2_label = path.name.replace(".nd2", "")
    arr = arr.expand_dims(Axes.REGION).assign_coords({Axes.REGION: [nd2_label]})

    # single-channel images don't include C
    if "C" not in arr.dims:
        arr = arr.expand_dims("C")
        channel = arr.metadata["metadata"].channels[0].channel.name.strip()
        arr = arr.assign_coords(C=[channel])
    else:
        # sanitize inputs that may contain leading/trailing spaces
        arr = arr.assign_coords(C=[channel.strip() for channel in arr.C.values])
    # single-field images don't include P
    if "P" not in arr.dims:
        arr = arr.expand_dims("P")
        point_coords = ["0"]
    else:
        point_coords = list(map(str, range(arr.P.size)))
    arr = arr.assign_coords(P=point_coords)
    rename_dict = dict(
        C=Axes.CHANNEL,
        P=Axes.FIELD,
        Y=Axes.Y,
        X=Axes.X,
    )
    if "T" in arr.dims:
        rename_dict["T"] = Axes.TIME
    if "Z" in arr.dims:
        rename_dict["Z"] = Axes.Z
    arr = arr.rename(rename_dict)

    return xr.Dataset(dict(intensity=arr))
