import numpy as np

def get_layout_indexing(scope_name: str, dimension: int) -> np.ndarray:
    """
    constructs an array whose elements index over montage captures for the given microscope and dimension.

    for example, flo uses snake-by-rows, right-and-down indexing for montages.
    So the array indexing a 3x3 mongage on flo is given by:

    1 2 3
    6 5 4
    7 8 9
    """
    default = np.arange(dimension**2).reshape(dimension,dimension) + 1
    snake_by_rows = default.copy()
    snake_by_rows[1::2] = default[1::2,::-1] # reverse every other row

    if scope_name.lower() == 'ixm':
        ''' 1 2 3
            4 5 6
            7 8 9 '''
        return default

    elif scope_name.lower() == 'flo':
        ''' 1 2 3
            6 5 4
            7 8 9 '''
        return snake_by_rows

    elif scope_name.lower() == 'flo2':
        ''' 3 4 9
            2 5 8
            1 6 7 '''
        return snake_by_rows.transpose()[::-1]

    elif scope_name.lower() in {'ds', 'ds2', 'ds3'}:
        ''' 7 8 9
            6 5 4
            1 2 3 '''
        return snake_by_rows[::-1]

    else:
        raise Exception("no matching microscope")
