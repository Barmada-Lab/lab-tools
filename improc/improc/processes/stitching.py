from operator import itemgetter
from typing import DefaultDict
from .common.utils import get_layout_indexing
from scipy.sparse import linalg

import numpy as np

def stitch(microscope: str, images: list[np.ndarray], t_o=0.1):
    #+TAG:DIRTY
    # Assumption made that stitched images all have square geometry.
    dim = int(np.sqrt(len(images)))
    layout = get_layout_indexing(microscope, dim)
    # Flatten layout sequence and get permuted sequence of paths.
    perm_paths = itemgetter(*(layout.reshape(-1)))(images)
    # Reshape images into correct 2D locations but with each image being a 1D array.
    imgarray = np.array(perm_paths)
    # Assuming all images are of same size, reshape into larger array with images in place.
    imgarray = imgarray.reshape((dim, dim, *imgarray[0].shape))

    # Used in later computations.
    img_size_y, img_size_x = imgarray[0, 0].shape

    '''
    #Same transform can be used for both x and y overlaps, which is tremendously useful
    #Equation form is very reminiscent of a system of difference equations, and I'm sure it could be formulated as such
    #Create transform matrix     0   1   2   3   4   5   6   7   8
    mat = np.matrix(np.array([  [2, -1,  0, -1,  0,  0,  0,  0,  0],
                                [0,  2, -1,  0, -1,  0,  0,  0,  0],
                                [0,  0,  1,  0,  0, -1,  0,  0,  0],
                                [0,  0,  0,  2, -1,  0, -1,  0,  0],
                                [0,  0,  0,  0,  2, -1,  0, -1,  0],
                                [0,  0,  0,  0,  0,  1,  0,  0, -1],
                                [0,  0,  0,  0,  0,  0,  1, -1,  0],
                                [0,  0,  0,  0,  0,  0,  0,  1, -1],
                                [0,  0,  0,  0,  0,  0,  0,  0,  0] ]))
    '''

    #Build transform matrix
    Y, X = imgarray.shape[0], imgarray.shape[1]
    dim = Y * X

    #Matrix full of zeros
    a = np.zeros((dim, dim), dtype=np.int8)

    #Increase every diagonal by one
    diag = np.diag_indices(dim)

    # 2 if the next k is less than one minus the last column?
    a[diag] = [2 if ( (k+1) < (Y-1)*X ) and ( (k+1) % X != 0 ) else 1 for k in range(dim)]

    def neighs(k):
        if (k+1) < (Y-1)*X and ((k+1) % X) != 0:
            return (k+1, k+X)
        elif (k+1) > (Y-1)*X:
            return (k+1,)
        else:
            return (k+X,)

    for k, row in enumerate(a[:-1]):
        for i in neighs(k):
            row[i] = -1

    a[-1, -1] = 0
    mat = np.matrix(a)

    #Could build auxiliary functions to determine whether in last row or last column -- would make code cleaner
    #Build overlap vectors
    xoverlaps = [t_o if (k+1) % Y != 0 else 0 for k in range(dim)]
    yoverlaps = [t_o if (k+1) <= (Y-1)*Y else 0 for k in range(dim)]

    xtrans = linalg.gmres(mat, xoverlaps)[0] * img_size_x
    ytrans = linalg.gmres(mat, yoverlaps)[0] * img_size_y

    #May also have to minimize values too..although, perhaps anotherway (shifting entire translation transform to another origin, as they are all relative)
    ##Yes, this is an affine transformation and origin is irrelevant
    #xtrans -= xtrans.max()
    #ytrans -= ytrans.max()

    xtrans -= xtrans[0]
    ytrans -= ytrans[0]

    #X DIFFERENCING MATRIX
    a = np.zeros((dim, dim), dtype=np.int8)
    for k, row in enumerate(a):
        if k % X == 0:
            row[k] = 1
        else:
            row[k-1], row[k] = 1, -1
    a[0,0] = 0

    xslices = np.dot(a, xtrans).astype(np.int32)

    #Y DIFFERENCING MATRIX
    a = np.zeros((dim, dim), dtype=np.int8)
    for k, row in enumerate(a[:]):
        if k < X:
            row[k] = 1
        else:
            row[k-X], row[k] = 1, -1

    a[0,0] = 0
    yslices = np.dot(a, ytrans).astype(np.int32)

    arr = imgarray


    #Shift tiles and slice based on translations
    slices = []
    for i in range(Y):
        for j in range(X):
            slices.append(arr[i, j][yslices[i*X + j]:, xslices[i*X + j]:])

    ##Make blank array to be filled
    stitched_arr = np.zeros((Y*img_size_y, X*img_size_x), dtype=np.uint16)
    running_col_totals = DefaultDict(int)
    running_row_totals = DefaultDict(int)
    for i, tile in enumerate(slices):
        tile_rows, tile_cols = tile.shape
        row = int(np.floor(i / X))
        col = i % X

        row_coord = running_row_totals[row]
        col_coord = running_col_totals[col]

        coords = (slice(col_coord, col_coord + tile_rows),
                  slice(row_coord, row_coord + tile_cols))

        stitched_arr[coords] = tile

        #Keep track of total pixels placed so next tile can be situated properly
        running_row_totals[row] += tile_cols
        running_col_totals[col] += tile_rows


    max_col = max(running_col_totals.values())
    max_row = max(running_row_totals.values())


    img = stitched_arr[:max_col, :max_row]
    # On special request, images taken via the 'ixm' need a 90 degree rotation.
    if microscope in {'ixm'}:
        img = np.rot90(img)

    return img
