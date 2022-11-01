import os
from collections import defaultdict
import numpy as np
from glob import glob
from skimage import io

def to8bit(stack):
    stack = stack.astype(float)
    min, max = np.min(stack), np.max(stack)
    stack -= min
    stack /= (max - min)
    stack *= 255
    return stack.astype(np.uint8)

def convertStacks(path, colorDirPairs):
    # Acquire set of all existing 16-bit stack names.
    print(f"looking in {path}")
    stackNames = set(os.path.basename(p) for p in glob(f'{path}/*/*tif'))

    for stackName in stackNames:
        # Attempt to load each color pair.
        toComposite = []
        for (color, directory) in colorDirPairs:
            stackPath = os.path.join(path, directory, stackName)

            # In the event that the color is paired with None, then set the stack to None as well.
            if directory is None:
                toComposite.append(None)
                continue

            try:
                stack = io.imread(stackPath)
            except FileNotFoundError:
                print(f'Could not find {stackPath}. Leaving stack blank.')
                toComposite.append(None)
                continue

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
#indir='processed_imgs/stacked', outdir='stacks'
def makePNGs(input_path, output_path, colorDirPairs):
    # Make directories to save PNGs and do so.
    os.makedirs(os.path.join(output_path), exist_ok=True)
    for stackName, toComposite in convertStacks(input_path, colorDirPairs):
        print(f"Converting {stackName} to PNG...")
        for ix, composite in enumerate(toComposite):
            savePath = os.path.join(output_path, stackName.split('.')[0] + f'_{str(ix)}.png')
            print(f"Saving to {savePath}")
            io.imsave(savePath, np.moveaxis(composite, 0, -1))

def findMaxStackLength(path):
    imagepaths = glob(os.path.join(path, "*.png"))
    # Image paths expected to be of type: */*_[0-9]+.png .
    return max(int(os.path.basename(fn).split('.')[0].split('_')[-1]) for fn in imagepaths) + 1

# Why does the client need this information? This is not good design. Should be redone.
def mkClientPathsData(path):
    # Map each well-id to a list of image paths.
    imagepaths = glob(os.path.join(path, '*.png'))
    idToPaths = defaultdict(list)

    for imagepath in imagepaths:
        # Remove path information before the input directory.
        # Image paths expected to be of type: */STACKID_*.png.
        stackId = '_'.join(os.path.basename(imagepath).split('_')[:-1])
        idToPaths[stackId].append(imagepath.replace(path,"./static/stacks/"))

    print(idToPaths)

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
