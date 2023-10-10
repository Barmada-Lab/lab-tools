#from cv2 import boundingRect
import os
import shutil
import struct
import tempfile
import zipfile
from collections import defaultdict
import json

def decode(roi_zip_path, outpath=''):
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

            print(roi_type)

            # This is an oval type. Will be taken to be a circle.
            if roi_type in {2, 8}: 
                x, y = left, top
                r = (right - left) / 2
                data = {'status':'alive', 'data':[{'x':int(x+r),'y':int(y+r), 'r':r}], 'id':ix, 'lastIxAlive':0}
                json.dump(data, open(f'{outpath}/{outpath}-{ix}.json', 'w'))
            #nCoords = struct.unpack('>h', data[16:18])
            #subType = struct.unpack('>h', data[48:50])
            #position = struct.unpack('>h', data[56:58])
            #print(subType)
            #for i in range(1, 81, 2):
                #print(i)
                #xCoords = struct.unpack('H', data[64 + i: 66 + i])
                #print(xCoords)
            #break

def encode(contour, name):
    #This should be temporary -- would like to think about storing contours in a different format
    contour = contour[:, 0, :]
    #Acquire bounding rectangle corner values necessary for roi encoding
    #x, y, w, h = boundingRect(contour)
    #top, left, bottom, right = y, x, y + h, x + w
    top, left, bottom, right = 0, 0, 0, 0#y + h, x + w
    #Acquire coordinates
    x, y = contour[:, 0], contour[:, 1]
    #Begin encoding
    num_coords = len(x) #Length of either x or y yields total number of coordinates
    roi_bytes = struct.pack('>4ch2B5h', b'I', b'o', b'u', b't', 225, 7, 0, top, left, bottom, right, num_coords)
    #Extracted this value by decoding current rois, not too clear on its use for extracting color
    #stroke_color = 4294901760L -- Python3 generated error due to 'L' suffix
    stroke_color = 4294901760
    #header2 is an extra set of parameters that can be specified and are placed after coordinates
    header2_offset = 64 + num_coords * 2 #Multiply by two since each short is currently two bytes
    roi_bytes += struct.pack('>4fh3I2h2Bh2I', 0.0, 0.0, 0.0, 0.0, 0, 0, stroke_color, 0, 0, 0, 0, 0, 0, 0, header2_offset)
    for xx in x:
        roi_bytes += struct.pack('>h', xx)
    for yy in y:
        roi_bytes += struct.pack('>h', yy)
    #Add header2, primarily to add roi name
    name_length = len(name)
    name_offset = (6 * 4) + (2) + (2 * 1) + (4) + (4) + (2 * 4) #Sum of bytes in header2
    roi_bytes += struct.pack('>6Ih2BIf2I', 0, 0, 0, 0, name_offset, name_length, 0, 0, 0, 0, 0.0, 0, 0)
    for c in name:
        roi_bytes += struct.pack('>h', ord(c))
    return roi_bytes 

def export_ij_rois(outpath, well, neurons, crop_val=0):
    '''This exportation method will adhere to the format present in the lab.'''
    ij_roi_dir = tempfile.mkdtemp('IJ_roi_series')
    tp_to_rois = defaultdict(list)
    for neuron in neurons:
        contours = [roi.contour + crop_val for roi in neuron.roi_series]
        for timepoint, contour in enumerate(contours):
            tp_to_rois[timepoint].append((neuron.ID, encode(contour, name=str(neuron.ID))))
    for timepoint, ij_rois in tp_to_rois.items():
        roi_paths = []
        for ij_roi in ij_rois:
            roi_path = os.path.join(ij_roi_dir, well + '_' + str(timepoint) + '_' + str(ij_roi[0]) + '.roi')
            with open(roi_path, 'wb') as f:
                f.write(ij_roi[1])
                roi_paths.append(roi_path)
        ij_rois_outpath = os.path.join(outpath, 'IJ_rois')
        os.makedirs(ij_rois_outpath, exist_ok=True)
        with zipfile.ZipFile(os.path.join(ij_rois_outpath, well + '-T' + str(timepoint+1) + '_RoiSet.zip'), 'w') as roi_zip:
            for roi_path in roi_paths:
                #Expected roi_path is *_[0-9]+.roi
                roi_ID = int(roi_path.split('_')[-1].split('.')[0]) + 1
                roi_zip.write(roi_path, str(roi_ID) + '.roi')
    shutil.rmtree(ij_roi_dir)
