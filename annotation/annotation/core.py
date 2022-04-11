import pathlib

from .constants import AUTO_ENC, JSON_ENC, PKL_ENC, IJ_ENC

from . import convert as vert
from . import utils

def convert(src: pathlib.Path, dest: pathlib.Path, src_encoding: str, dest_encoding: str):
    if src_encoding == AUTO_ENC:
        if (encoding := utils.determine_file_encoding(src.name)) != None:
            src_encoding = encoding
        else:
            raise Exception(f"Unknown encoding for file {src.name}")

    if [src_encoding, dest_encoding] == [JSON_ENC, IJ_ENC]:
        vert.json_to_ij(src, dest)
    else:
        raise Exception(f"Unsupported conversion between {src_encoding} and {dest_encoding}")
