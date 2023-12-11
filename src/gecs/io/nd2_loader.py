import pathlib as pl

import nd2

def read_nd2(path: pl.Path):

    arr = nd2.imread(path, dask=True, xarray=True)
    nd2_label = path.name.replace(".nd2","")
    arr = arr.expand_dims("region").assign_coords(region=[nd2_label])

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
        C="channel",
        P="field",
        Y="y",
        X="x",
    )
    if "T" in arr.dims:
        rename_dict["T"] = "t"
    if "Z" in arr.dims:
        rename_dict["Z"] = "z"
    arr = arr.rename(rename_dict)
    return arr