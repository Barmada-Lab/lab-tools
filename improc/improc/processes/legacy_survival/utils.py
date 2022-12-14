import os
from collections import defaultdict

import cv2
import tifffile
import numpy as np
from scipy import optimize

from . import legacy as transforms

def comp_cent(contour):
    mom = cv2.moments(contour)
    return np.array([mom['m10'] / mom['m00'], mom['m01'] / mom['m00']], dtype=np.uint32)

def comp_lsq_circle(contour):
    '''Contour expected to be in OpenCV format'''
    contour = contour[:, 0]
    x, y = contour[:, 0], contour[:, 1]
    centroid = comp_cent(contour)
    cx, cy = centroid[0], centroid[1]
    theta, r = np.arctan2(y-cy, x-cx), np.hypot(x-cx, y-cy)
    theta += np.pi
    rr = optimize.leastsq(lambda rr: sum((rr - r) ** 2), x0=.5)[0]
    mse = np.abs(np.mean((rr - r) ** 2))
    return float(mse)

def find_hulls(img):
    img = np.copy(img)
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [cv2.convexHull(contour) for contour in contours]
    return hulls
   
#Do not appreciate fact that an empty list can sneak in to the tuple
#Size threshold is not in use
def find_hulls_with_inner_contours(img, size_threshold=50):
   img = np.copy(img)
   hulls_inners = []
   _, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
   if not contours: return hulls_inners
   hierarchy = hierarchy[0]
   for ix, contour in enumerate(contours):
       #Ensure this is outer contour of sufficient size
       if hierarchy[ix][3] != -1 or cv2.contourArea(contour) < size_threshold:
           continue
       #Now acquire corresponding inner contours, if they exist
       inner_contours = []
       #Index of first child contour, will be -1 if does not exist
       child_ix = hierarchy[ix][2]
       while child_ix != -1:
           inner_contours.append(contours[child_ix])
           #Set child index to the next index to check for more children
           child_ix = hierarchy[child_ix][0]
       hulls_inners.append((cv2.convexHull(contour), inner_contours))
   return hulls_inners

def nonzero_percentile_threshold(img, percentile):
    if img.max() != 0: 
        img[img <= np.percentile(img[img != 0], percentile)] = 0

class ImageLogger:
    '''
    Enables saving images in a hierarchy of phases and sequences. Each phase is registered prior to use.
    Phases can be switched by setting logger to a particular phase. Images corresponding to steps within a
    phase's sequence are recorded. The order of these steps is recorded.
    '''

    def __init__(self, logpath):
        self.phases = defaultdict(list)
        self.logpath = logpath
        os.makedirs(self.logpath, exist_ok=True)

    def find_then_box_rois(self, img):
        # Convert to 8-bit.
        img = transforms.to_8bit(img)

        # Find contours.
        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert to rgb to draw color on.
        img = transforms.to_rgb(img)

        # For each contour, draw a rectangle around its bounding box.
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        return img

    def draw_slices(self, img, slices):
        # Convert to rgb to draw color on.
        img = transforms.to_rgb(img)

        for s in slices:
            x, w = s[1].start, s[1].stop - s[1].start
            y, h = s[0].start, s[0].stop - s[0].start
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        return img

    def draw_contours(self, img, contours):
        # Convert to rgb to draw color on.
        img = transforms.to_rgb(img)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
        return img

    def set_image_name(self, name):
        self.imgname = str(name)

        os.makedirs(os.path.join(self.logpath, name), exist_ok=True)

    def set_timepoint(self, timepoint):
        self.timepoint = str(timepoint)

        # Reset phases as this is a new sequence.
        self.phases = defaultdict(list)

    # Can later provide another option for box or contour.
    def log(self, phase, step, img, box=False, slices=None, contours=None):
        # Ensure that working on a copied image and no changes to input image are made.
        outpath = os.path.join(self.logpath, self.imgname, phase)

        if box: img = self.find_then_box_rois(img)
        elif slices is not None:
            img = self.draw_slices(img, slices)
        elif contours is not None:
            img = self.draw_contours(img, contours)

        # Make requisite directories if necessary.
        os.makedirs(outpath, exist_ok=True)

        # Add the step to this phase.
        self.phases[phase].append(step)

        # Suffix the image name with number indicating its order in the sequence.
        step_num = str(len(self.phases[phase]))
        outname = f'T{self.timepoint}-S{step_num}-{step}-{self.imgname}.tif'

        # Log image.
        tifffile.imsave(os.path.join(outpath, outname), img)
