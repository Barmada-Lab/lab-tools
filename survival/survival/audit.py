from pathlib import Path
import imageio

from skimage import exposure, segmentation # type: ignore
from PIL import Image
import numpy as np
import cv2

def composite_segmented_frame(raw: np.ndarray, segmented: np.ndarray):
    rescaled = exposure.rescale_intensity(raw, out_range='uint8')
    marked = segmentation.mark_boundaries(rescaled, segmented)
    marked = exposure.rescale_intensity(marked, out_range='uint8')
    return marked

def composite_segmented_frames(raw_frames: np.ndarray, segmented_frames: np.ndarray):
    return np.array([composite_segmented_frame(raw, segmented) for raw, segmented in zip(raw_frames, segmented_frames)])

def write_audited_segmentation_gif(raw_frames: np.ndarray, segmented_frames: np.ndarray, path: Path):
    audited_frames = composite_segmented_frames(raw_frames, segmented_frames)
    _, h, w, _ = audited_frames.shape
    frame_0 = Image.fromarray(audited_frames[0])
    frame_0.save(path, format='GIF', save_all=True, 
        append_images=[Image.fromarray(frame) for frame in audited_frames[1:]], duration=500, loop=0)