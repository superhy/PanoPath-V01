'''
@author: Yang Hu

*reference the code from: https://github.com/deroneriksson/python-wsi-preprocessing
'''

import sys

from PIL import Image
import cv2

import numpy as np
from support.tools import Time
from support.tools import np_info
from skimage.color import rgb2hed, hed2rgb

Image.MAX_IMAGE_PIXELS = None
sys.path.append("..")

def convert_rgb_to_bgr(img_rgb):
    '''
    for cv2 APIs, need to convert rgb image (default in PIL) to bgr color format
    '''
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr

def convert_rgb_to_bgr_byhand(img_rgb):
    img_bgr = img_rgb[...,::-1]
    return img_bgr


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.
    
    Args:
      np_img: The image represented as a NumPy array.
    
    Returns:
       The NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64" or np_img.dtype == "float16":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)

def pil_rgb_2_ihc_dab(ihc_pil_img):
    '''
    transfer the pil_img to DAB(brown) channel,
    should be only used for ihc stained slides(tiles)
    
    Ref: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_ihc_color_separation.html
    '''
    # pil -> nparray
    ihc_rgb = np.array(ihc_pil_img)
    # transfer to IHC HEB space
    ihc_hed = rgb2hed(ihc_rgb)
    
    # filtering to only have DAB channel
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_dab = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    # scale the data range back to 0~255 UTF-8
    ihc_dab = np.clip(ihc_dab, 0, 1)
    ihc_dab = (ihc_dab * 255).astype(np.uint8)

    # nparray -> pil
    return Image.fromarray(ihc_dab)

def pil_to_np_rgb(pil_img, show_np_info=False):
    """
    Convert a PIL Image to a NumPy array.
    
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).
    
    Args:
      pil_img: The PIL Image.
    
    Returns:
      The PIL image converted to a NumPy array.
    """
    t = Time()
    rgb = np.asarray(pil_img)
    if show_np_info == True:
        np_info(rgb, "RGB", t.elapsed())
        
    return rgb

def mask_rgb(rgb, mask, show_np_info=False):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
    
    Args:
      rgb: RGB image as a NumPy array.
      mask: An image mask to determine which pixels in the original image should be displayed.
    
    Returns:
      NumPy array representing an RGB image with mask applied.
    """
    t = Time()
    result = rgb * np.dstack([mask, mask, mask])
    if show_np_info == True:
        np_info(result, "Mask RGB", t.elapsed())
        
    return result

def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
    
    Args:
      np_img: Image as a NumPy array.
    
    Returns:
      The percentage of the NumPy array that is masked.
    """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage

if __name__ == '__main__':
    pass