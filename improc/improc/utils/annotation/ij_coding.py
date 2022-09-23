import struct
import zipfile
import json
import os

from .types import Roi

def encode(roi: Roi, id: str) -> bytes:
    #+TAG:DIRTY
    #Acquire bounding rectangle corner values necessary for roi encoding
    #x, y, w, h = boundingRect(contour)
    #top, left, bottom, right = y, x, y + h, x + w
    top, left, bottom, right = 0, 0, 0, 0 # y + h, x + w
    #Acquire coordinates
    xs, ys = roi.contour[:, 0], roi.contour[:, 1]
    #Begin encoding
    num_coords = len(xs) #Length of either x or y yields total number of coordinates
    roi_bytes = struct.pack('>4ch2B5h', b'I', b'o', b'u', b't', 225, 7, 0, top, left, bottom, right, num_coords)
    #Extracted this value by decoding current rois, not too clear on its use for extracting color
    #stroke_color = 4294901760L -- Python3 generated error due to 'L' suffix
    stroke_color = 4294901760
    #header2 is an extra set of parameters that can be specified and are placed after coordinates
    header2_offset = 64 + num_coords * 2 #Multiply by two since each short is currently two bytes
    roi_bytes += struct.pack('>4fh3I2h2Bh2I', 0.0, 0.0, 0.0, 0.0, 0, 0, stroke_color, 0, 0, 0, 0, 0, 0, 0, header2_offset)
    for x in xs:
        roi_bytes += struct.pack('>h', x)
    for y in ys:
        roi_bytes += struct.pack('>h', y)
    #Add header2, primarily to add roi name
    name = id
    name_length = len(name)
    name_offset = (6 * 4) + (2) + (2 * 1) + (4) + (4) + (2 * 4) #Sum of bytes in header2
    roi_bytes += struct.pack('>6Ih2BIf2I', 0, 0, 0, 0, name_offset, name_length, 0, 0, 0, 0, 0.0, 0, 0)
    for c in name:
        roi_bytes += struct.pack('>h', ord(c))
    return roi_bytes

def decode(roi_zip_path, outpath=''):
    #+TAG: DIRTY
    with zipfile.ZipFile(roi_zip_path) as roi_zip:
        fnames = roi_zip.namelist()
        for ix, fname in enumerate(fnames):
            data = roi_zip.read(fname)
            version = struct.unpack('>B', data[5:6])
            roi_type = int(struct.unpack('>B', data[6:7])[0])
            top = struct.unpack('>h', data[8:10])[0]
            left = struct.unpack('>h', data[10:12])[0]
            bottom = struct.unpack('>h', data[12:14])[0]
            right = struct.unpack('>h', data[14:16])[0]

            # Ensure output directory exists.
            os.makedirs(outpath, exist_ok=True)

            # This is an oval type. Will be taken to be a circle.
            if roi_type in {2, 8}:
                x, y = left, top
                r = (right - left) / 2
                data = {'status':'alive', 'data':[{'x':int(x+r),'y':int(y+r), 'r':r}], 'id':ix, 'lastIxAlive':0}
                json.dump(data, open(f'{outpath}/{outpath}-{ix}.json', 'w'))
