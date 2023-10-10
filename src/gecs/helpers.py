from pathlib import Path

def img_path_loader(paths: list[Path]) -> list[Path]:
    """ 
    Provides a consistent interface for specifying images either from a directory 
    or a list of files. 
    """
    if len(paths) == 1 and paths[0].is_dir():
        paths = list(paths[0].glob("*"))

    return [path for path in paths if path.is_file()]