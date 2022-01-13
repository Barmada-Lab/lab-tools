from glob import glob
from string import ascii_uppercase

import subprocess
import random
import json
import sys
import os

from . import util

def deploy(args):
    input_path = args["path"]
    scramble = args["scramble"]
    expName = args["exp_name"]
    masaId = args["masa_name"]
    masa_path = os.path.join(args["data_loc"], masaId)

    # Associate certain known directory names within channels.
    channels = [os.path.basename(p).lower() for p in glob(f'{input_path}/processed_imgs/stacked/*')]
    print('channels:', channels)
    colorDirPairs = [('red', 'white_light'), ('green', 'white_light'), ('blue', 'white_light')]

    redCh, greenCh, blueCh = None, None, None
    if 'rfp' in channels:
        colorDirPairs[0] = ('red', channels.pop(channels.index('rfp')).upper())
    elif 'tritc' in channels:
        colorDirPairs[0] = ('red', channels.pop(channels.index('tritc')).upper())
    if 'gfp' in channels:
        colorDirPairs[1] = ('green', channels.pop(channels.index('gfp')).upper())
    if 'dapi' in channels:
        colorDirPairs[2] = ('blue', channels.pop(channels.index('dapi')).upper())
    elif 'white_light' in channels:
        colorDirPairs[2] = ('blue', channels.pop(channels.index('white_light')))

    # If there are any remaining channels, associate them with any non-used colors.
    for ix, (color, dir) in enumerate(colorDirPairs):
        if dir is None:
            if len(channels) > 0:
                colorDirPairs[ix] = (color, channels.pop())
            else:
                break

    print("converting PNGs...")
    # Create data.
    tif_path = os.path.join(input_path,"processed_imgs","stacked")
    png_path = os.path.join(masa_path, "data", "stacks")
    util.makePNGs(tif_path, png_path, colorDirPairs)
    print("done converting PNGs.")

    # Is this used at all?
    colors = ['gray', 'red', 'green', 'blue']

    roidir = f'/{expName}-rois'

    # Search the client image filenames for the maximum stack length in the dataset.
    maxStackLength = util.findMaxStackLength(input_path)
    stackNames, stackIdMap = util.mkClientPathsData(input_path)
    stackIds = util.mkServerPathsData(stackNames)
    serverInput = {  'expName' : expName,
                     'stackdir' : '/static/stacks',
                     'roidir' : roidir,
                     'stackIds' : stackIds,
                     'port': 6000 + int(masaId[4:])
                  }

    clientInput = { 'expName' : expName,
                    'masaId' : masaId,
                    'colors' : colors,
                    'maxStackLength' : maxStackLength,
                    'stackNames' : stackNames,
                    'totalStackNum' : len(stackNames),
                    'stackIdMap' : stackIdMap,
                  }

    if scramble:
        # Acquire double capital letter prefixes (e.g., AA) for each stack.
        letters = [l1 + l2 for l1 in ascii_uppercase for l2 in ascii_uppercase]
        random.shuffle(letters)
        letters = letters[:len(stackNames)]

        # Randomize stack name order.
        random.shuffle(stackNames)

        scrambledKeys = dict()
        for ix, stackName in enumerate(stackNames):
            scrambledKeys[stackName['id']] = letters[ix]
            stackName['id'] = letters[ix]

        stackIdMap = dict()
        for ix, stackName in enumerate(stackNames):
            stackIdMap[stackName['id']] = ix

        scrambledDict = {'scrambledKeys': scrambledKeys, 'stackIdMap': stackIdMap}

        clientInput.update(scrambledDict)


    data_path = os.path.join(masa_path, "server")
    with open(os.path.join(data_path, "serverSetup.json"), 'w') as f:
        json.dump(serverInput, f)

    with open(os.path.join(data_path, "clientSetup.json"), 'w') as f:
        json.dump(clientInput, f)

    other_server_path = os.path.join(args["server_loc"], masaId)
    with open(os.path.join(other_server_path, "src", "serverSetup.json"), 'w') as f:
        json.dump(serverInput, f)

    with open(os.path.join(other_server_path, "src", "clientSetup.json"), 'w') as f:
        json.dump(clientInput, f)

    print("Running npm build...")
    subprocess.run(["npm", "run-script", "build"],
                   cwd=other_server_path)

    subprocess.run(["python", "refresh-exp-names.py"],
                   cwd="/data/masa/")
