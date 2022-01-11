import json
import os
import random
import sys
from glob import glob
from collections import defaultdict
from string import ascii_uppercase
import numpy as np
from scipy.misc import imsave as imsave
import skimage.external.tifffile as tifffile

def to8bit(stack):
    stack = stack.astype(float)
    min, max = np.min(stack), np.max(stack)
    stack -= min
    stack /= (max - min)
    stack *= 255 
    return stack.astype(np.uint8)

def convertStacks(path, colorDirPairs):
    # Acquire set of all existing 16-bit stack names.
    stackNames = set(os.path.basename(p) for p in glob(f'{path}/*/*tif'))

    colorPairs = [('red', 'tritc'), ('green', 'gfp'), ('blue', 'white_light')] 

    for stackName in stackNames:
        # Attempt to load each color pair.
        toComposite = []
        for (color, dir) in colorDirPairs:
            stackPath = f'{path}/{dir}/{stackName}'

            # In the event that the color is paired with None, then set the stack to None as well.
            if dir is None:
                toComposite.append(None)
                continue

            try: 
                stack = tifffile.imread(stackPath)
                print('SHAPE', stack.shape)
            except FileNotFoundError:
                print(f'Could not find {stackPath}. Leaving stack blank.')
                toComposite.append(None)

            else:
                # Convert to 8-bit and set for compositing.
                stack = to8bit(stack)
                # Check if stack is 2D (a single image) and make 3D.
                if len(stack.shape) == 2: stack = np.expand_dims(stack, axis=0)
                toComposite.append(stack)

        # NOTE: Shapes can be different. If this becomes a problem, will need to handle here.

        # Acquire shape of an extant image.
        shape = [stack.shape for stack in toComposite if stack is not None][0]

        # If two stacks were set to None, then only one channel exists. Copy this channel to the 
        # others so it is set to gray (R = G = B).
        if sum([1 for s in toComposite if s is None]) == 2:
            extant = list(filter(lambda s: s is not None, toComposite))[0]
            for ix, stack in enumerate(toComposite):
                if stack is None:
                    toComposite[ix] = extant

        # Check if any single stacks set to None and set them to blank with right shape.
        # If a single color does not exist, the remaining two colors will still exist.
        for ix, stack in enumerate(toComposite):
            if stack is None:
                toComposite[ix] = np.zeros(shape)

        yield stackName, np.stack(toComposite, axis=1)

def makePNGs(path, colorDirPairs, indir='./processed_imgs/stacked', outdir='./static/stacks'):
    # Make directories to save PNGs and do so.
    os.makedirs(os.path.join(path, outdir), exist_ok=True)
    for stackName, toComposite in convertStacks(os.path.join(path, indir), colorDirPairs):
        print(stackName)
        #if os.path.isfile(f'm:/ns/kathaniel92/static/stacks/{stackName.split(".")[0]}_23.png'):
        #    continue
        for ix, composite in enumerate(toComposite):
            savePath = os.path.join(path, outdir, stackName.split('.')[0] + f'_{str(ix)}.png')
            imsave(savePath, np.moveaxis(composite, 0, -1))

def findMaxStackLength(path, indir='./static/stacks'):
    imagepaths = glob(f'{os.path.join(path, indir)}/*png')
    # Image paths expected to be of type: */*_[0-9]+.png . 
    return max(int(os.path.basename(fn).split('.')[0].split('_')[-1]) for fn in imagepaths) + 1

# Why does the client need this information? This is not good design. Should be redone.
def mkClientPathsData(path, indir='./static/stacks'):
    # Map each well-id to a list of image paths.
    imagepaths = glob(f'{os.path.join(path, indir)}/*png')
    idToPaths = defaultdict(list)

    for imagepath in imagepaths: 
        # Remove path information before the input directory.
        imagepath = imagepath[imagepath.find(indir):]
        # Image paths expected to be of type: */STACKID_*.png.
        stackId = '_'.join(os.path.basename(imagepath).split('_')[:-1])
        idToPaths[stackId].append(imagepath)

    # Sort each image path list to ensure they are traversed in order.
    for id, paths in idToPaths.items():
        idToPaths[id] = sorted(paths, key=lambda p : int(os.path.basename(p).split('.')[0].split('_')[-1]))

    # Prepare a list of dicts containing this data. 
    data = [{'id' : stackId, 'fileNames' : paths} for stackId, paths in idToPaths.items()]

    # A well-id may have underscores. Typical well-ids are of form [A-Z][0-9]+, e.g., A01, but well-ids such
    # as A01_1 have also been used (the corresponding png paths would be A01_1_[0-9]+ , in this case). The sorting
    # procedure here handles an arbitrary amount of underscore delimiters. It makes the assumption that all 
    # file names will have the same quantity of underscores, and that each term delimited by underscores, except
    # the first, is an integer.

    total_terms = len(data[0]['id'].split('_'))
    for i in range(1, total_terms):
        # It is necessary by the least specific integers first, and then work inwards, so that the final result
        # is correct.
        data.sort(key=lambda d : int(d['id'].split('_')[total_terms - i]))

    # Sort by the first number (e.g., the 01 in A01
    #data.sort(key=lambda d : int(d['id'].split('_')[0][1:]))

    # Finally sort by the first letter (e.g., the A in A01)
    #data.sort(key=lambda d : d['id'][0])

    # Make a map between the well-id and the list index. This is used when a user wishes to select a well-id by
    # name.
    dataMap = {datum['id'] : ix for ix, datum in enumerate(data)}

    return data, dataMap

def mkServerPathsData(clientPaths):
    return [d['id'] for d in clientPaths]

def run(path, scramble, expName, masaId):
    # Done for rapid testing.
    print(path, scramble)
    if path is None: path = 'L:/kathaniels/kathaniel82'

    # Associate certain known directory names within channels.
    channels = [os.path.basename(p).lower() for p in glob(f'{path}/processed_imgs/stacked/*')]
    print('channels:', channels)
    colorDirPairs = [('red', 'white_light'), ('green', 'white_light'), ('blue', 'white_light')]
    
    redCh, greenCh, blueCh = None, None, None
    if 'rfp' in channels:
        colorDirPairs[0] = ('red', channels.pop(channels.index('rfp')))
    elif 'tritc' in channels:
        colorDirPairs[0] = ('red', channels.pop(channels.index('tritc')))
    if 'gfp' in channels:
        colorDirPairs[1] = ('green', channels.pop(channels.index('gfp')))
    if 'dapi' in channels:
        colorDirPairs[2] = ('blue', channels.pop(channels.index('dapi')))
    elif 'white_light' in channels:
        colorDirPairs[2] = ('blue', channels.pop(channels.index('white_light')))

    # If there are any remaining channels, associate them with any non-used colors.
    for ix, (color, dir) in enumerate(colorDirPairs):
        if dir is None:
            if len(channels) > 0:
                colorDirPairs[ix] = (color, channels.pop())
            else: 
                break

    # Create data.
    makePNGs(path, colorDirPairs)

    # Is this used at all?
    colors = ['gray', 'red', 'green', 'blue']

    roidir = f'/{expName}-rois'

    # Search the client image filenames for the maximum stack length in the dataset.
    maxStackLength = findMaxStackLength(path)
    stackNames, stackIdMap = mkClientPathsData(path)
    stackIds = mkServerPathsData(stackNames)

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

    # If the setup is to be scrambled then update the client's input.
    if scramble:
        clientInput.update(scrambledDict)

    print(path)
    with open(f'{path}/serverSetup.json', 'w') as f:
        json.dump(serverInput, f)

    with open(f'{path}/clientSetup.json', 'w') as f:
        json.dump(clientInput, f)

if __name__ == '__main__':
    #path = f'n:/am/am182 mips'
    scramble = False
    masaId = sys.argv[1]
    path = sys.argv[2]
    expName = sys.argv[3]

    run(path, scramble, expName, masaId)
