'''

@author: yang hu

*reference the code from: https://github.com/deroneriksson/python-wsi-preprocessing
'''

import json
import math
import os
from pathlib import Path
import sys

import cv2
from numpy import dtype
from skimage import morphology
from skimage import filters
from skimage import feature
import scipy.ndimage.morphology as sc_morph

import numpy as np
from support import tools
from support.env import ENV
from support.tools import Time
from wsi import image_tools
from wsi.image_tools import mask_percent
from wsi.slide_tools import original_slide_and_scaled_pil_image


sys.path.append("..")




def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool",
                         show_np_info=False, print_over_info=False):
    """
    Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
    and eosin are purplish and pinkish, which do not have much green to them.
    
    Args:
      np_img: RGB image as a NumPy array.
      green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
      avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
      overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
      output_type: Type of array to return (bool, float, or uint8).
    
    Returns:
      NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
    """
    t = Time()
    
    g = np_img[:,:, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        if print_over_info == True:
            print(
              "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
                mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
        gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
    np_img = gr_ch_mask
    
    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255
    
    if show_np_info == True:
        tools.np_info(np_img, "Filter Green Channel", t.elapsed())
    
    return np_img

def filter_rgb_to_grayscale(np_img, output_type="uint8"):
    """
    Convert an RGB NumPy array to a grayscale NumPy array.
    Shape (h, w, c) to (h, w).
    Args:
      np_img: RGB Image as a NumPy array.
      output_type: Type of array to return (float or uint8)
    Returns:
      Grayscale image as NumPy array with shape (h, w).
    """
    # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
    grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
    if output_type != "float":
        grayscale = grayscale.astype("uint8")
    return grayscale

def filter_complement(np_img, output_type="uint8"):
    """
    Obtain the complement of an image as a NumPy array.
    Args:
      np_img: Image as a NumPy array.
      type: Type of array to return (float or uint8).
    Returns:
      Complement image as Numpy array.
    """
    if output_type == "float":
        complement = 1.0 - np_img
    else:
        complement = 255 - np_img
    return complement

def filter_hysteresis_threshold(np_img, low=50, high=100, output_type="uint8"):
    """
    Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.
    Args:
      np_img: Image as a NumPy array.
      low: Low threshold.
      high: High threshold.
      output_type: Type of array to return (bool, float, or uint8).
    Returns:
      NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
    """
    hyst = filters.apply_hysteresis_threshold(np_img, low, high)
    if output_type == "bool":
        pass
    elif output_type == "float":
        hyst = hyst.astype(float)
    else:
        hyst = (255 * hyst).astype("uint8")
    return hyst

def filter_otsu_threshold(np_img, output_type="uint8"):
    """
    Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.
    Args:
      np_img: Image as a NumPy array.
      output_type: Type of array to return (bool, float, or uint8).
    Returns:
      NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
    """
    otsu_thresh_value = filters.threshold_otsu(np_img)
    otsu = (np_img > otsu_thresh_value)
    if output_type == "bool":
        pass
    elif output_type == "float":
        otsu = otsu.astype(float)
    else:
        otsu = otsu.astype("uint8") * 255
    return otsu

def filter_canny(np_img, sigma=1, low_threshold=0, high_threshold=25, output_type="uint8"):
    """
    Filter image based on Canny algorithm edges.
    Args:
      np_img: Image as a NumPy array.
      sigma: Width (std dev) of Gaussian.
      low_threshold: Low hysteresis threshold value.
      high_threshold: High hysteresis threshold value.
      output_type: Type of array to return (bool, float, or uint8).
    Returns:
      NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
    """
    can = feature.canny(np_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    if output_type == "bool":
        pass
    elif output_type == "float":
        can = can.astype(float)
    else:
        can = can.astype("uint8") * 255
    return can

def filter_grays(rgb, tolerance=15, output_type="bool", show_np_info=False):
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.
    
    Args:
      np_img: RGB image as a NumPy array.
      tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
      output_type: Type of array to return (bool, float, or uint8).
    
    Returns:
      NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    t = Time()
    (h, w, c) = rgb.shape
    
    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:,:, 0] - rgb[:,:, 1]) <= tolerance
    rb_diff = abs(rgb[:,:, 0] - rgb[:,:, 2]) <= tolerance
    gb_diff = abs(rgb[:,:, 1] - rgb[:,:, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)
    
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if show_np_info == True:
        tools.np_info(result, "Filter Grays", t.elapsed())
        
    return result

def filter_l_r_half(rgb, direction='right', output_type="bool"):
    """
    filtering left or right half
    
    Args:
        rgb:
        direction: keep 'left' or 'right' part of the slide, discard the other part
    """
    (h, w, c) = rgb.shape
    result = np.ones((h, w))
    if direction == 'right':
        result[:, :int(w * 0.5)] = 0
    else:
        result[:, int(w * 0.5) + 1:] = 0
    
    if output_type == "bool":
        result = result.astype(bool)
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
        
    return result

def filter_leftmost_piece(rgb):
    """
    filtering left most piece from the image (usually it will be the control tissue)
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
    
    result = np.ones_like(gray, dtype=bool)
    
    if num_labels <= 2:
        return result
    
    leftmost_x = min(stats[1:, cv2.CC_STAT_LEFT])
    width_of_leftmost = max(stats[1:, cv2.CC_STAT_WIDTH])
    
    result[:, leftmost_x:leftmost_x + width_of_leftmost] = False
    return result

def filter_binary_dilation(np_img, disk_size=5, iterations=1, output_type="uint8"):
    """
    Dilate a binary object (bool, float, or uint8).
    Args:
      np_img: Binary image as a NumPy array.
      disk_size: Radius of the disk structuring element used for dilation.
      iterations: How many times to repeat the dilation.
      output_type: Type of array to return (bool, float, or uint8).
    Returns:
      NumPy array (bool, float, or uint8) where edges have been dilated.
    """
    if np_img.dtype == "uint8":
        np_img = np_img / 255
    result = sc_morph.binary_dilation(np_img, morphology.disk(disk_size), iterations=iterations)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result

def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool"):
    """
    Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
    red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.
    
    Args:
      rgb: RGB image as a NumPy array.
      red_lower_thresh: Red channel lower threshold value.
      green_upper_thresh: Green channel upper threshold value.
      blue_upper_thresh: Blue channel upper threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      display_np_info: If True, display NumPy array info and filter time.
    
    Returns:
      NumPy array representing the mask.
    """
    r = rgb[:,:, 0] > red_lower_thresh
    g = rgb[:,:, 1] < green_upper_thresh
    b = rgb[:,:, 2] < blue_upper_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
        
    return result


def filter_red_pen(rgb, output_type="bool", show_np_info=False):
    """
    Create a mask to filter out red pen marks from a slide.
    
    Args:
      rgb: RGB image as a NumPy array.
      output_type: Type of array to return (bool, float, or uint8).
    
    Returns:
      NumPy array representing the mask.
    """
    t = Time()
    result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
             filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
             filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
             filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
             filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
             filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
             filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
             filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
             filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if show_np_info == True:
        tools.np_info(result, "Filter Red Pen", t.elapsed())
    return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool"):
    """
    Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
    red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
    Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
    lower threshold value rather than a blue channel upper threshold value.
    
    Args:
      rgb: RGB image as a NumPy array.
      red_upper_thresh: Red channel upper threshold value.
      green_lower_thresh: Green channel lower threshold value.
      blue_lower_thresh: Blue channel lower threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      display_np_info: If True, display NumPy array info and filter time.
    
    Returns:
      NumPy array representing the mask.
    """
    r = rgb[:,:, 0] < red_upper_thresh
    g = rgb[:,:, 1] > green_lower_thresh
    b = rgb[:,:, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255

    return result


def filter_green_pen(rgb, output_type="bool", show_np_info=False):
    """
    Create a mask to filter out green pen marks from a slide.
    
    Args:
      rgb: RGB image as a NumPy array.
      output_type: Type of array to return (bool, float, or uint8).
    
    Returns:
      NumPy array representing the mask.
    """
    t = Time()
    result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
             filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
             filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
             filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
             filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
             filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
             filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
             filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
             filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
             filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
             filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
             filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
             filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
             filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
             filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if show_np_info == True:
        tools.np_info(result, "Filter Red Pen", t.elapsed()) 
        
    return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool"):
    """
    Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
    red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.
    
    Args:
      rgb: RGB image as a NumPy array.
      red_upper_thresh: Red channel upper threshold value.
      green_upper_thresh: Green channel upper threshold value.
      blue_lower_thresh: Blue channel lower threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      display_np_info: If True, display NumPy array info and filter time.
    
    Returns:
      NumPy array representing the mask.
    """
    
    r = rgb[:,:, 0] < red_upper_thresh
    g = rgb[:,:, 1] < green_upper_thresh
    b = rgb[:,:, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
        
    return result


def filter_blue_pen(rgb, output_type="bool", show_np_info=False):
    """
    Create a mask to filter out blue pen marks from a slide.
    
    Args:
      rgb: RGB image as a NumPy array.
      output_type: Type of array to return (bool, float, or uint8).
    
    Returns:
      NumPy array representing the mask.
    """
    t = Time()
    result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
             filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
             filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
             filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
             filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
             filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
             filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
             filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
             filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
             filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
             filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
             filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if show_np_info == True:
        tools.np_info(result, "Filter Red Pen", t.elapsed())
        
    return result


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8",
                                show_np_info=False, print_over_info=False):
    """
    Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
    is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
    reduce the amount of masking that this filter performs.
    
    Args:
      np_img: Image as a NumPy array of type bool.
      min_size: Minimum size of small object to remove.
      avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
      overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
      output_type: Type of array to return (bool, float, or uint8).
    
    Returns:
      NumPy array (bool, float, or uint8).
    """
    t = Time()
    
    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = min_size / 2
        if print_over_info == True:
            print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
              mask_percentage, overmask_thresh, min_size, new_min_size))
        rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type,
                                             print_over_info=print_over_info)
    np_img = rem_sm
    
    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255
        
    if show_np_info == True:
        tools.np_info(np_img, "Filter Red Pen", t.elapsed())
        
    return np_img

# def mask_percent(np_img):
#     """
#     Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
#     
#     Args:
#       np_img: Image as a NumPy array.
#     
#     Returns:
#       The percentage of the NumPy array that is masked.
#     """
#     if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
#         np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
#         mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
#     else:
#         mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
#         
#     return mask_percentage

def tissue_percent(np_img):
    """
    Determine the percentage of a NumPy array that is tissue (not masked).
    
    Args:
      np_img: Image as a NumPy array.
    
    Returns:
      The percentage of the NumPy array that is tissue.
    """
    return 100 - mask_percent(np_img)


def apply_image_filters_cd45(np_img, tumor_region_jsonpath=None, tumor_or_background=True, print_info=True):
    """
    Apply filters to image as NumPy array and optionally save and/or display filtered images. For IHC-CD45 staining
    
    Args:
      np_img: Image as NumPy array.
      tumor_or_background: True: the filter only left tumor area; False: the filter only left background area
      
      @attention: removed
      <slide_num=None, info=None, save=False, display=False>
      
          slide_num: The slide number (used for saving/displaying).
          info: Dictionary of slide information (used for HTML display).
          save: If True, save image.
          display: If True, display image.
    
    Returns:
      Resulting filtered image as a NumPy array.
    """
    np_rgb = np_img
    
    if print_info == True:
        print('Filter noise of various colors and objects that are too small')
        
    mask_not_green = filter_green_channel(np_rgb, print_over_info=print_info)
    mask_not_gray = filter_grays(np_rgb, tolerance=8)
    # mask_right = filter_l_r_half(np_rgb, direction='right')
    mask_right = filter_leftmost_piece(np_rgb)
    
    mask_no_red_pen = filter_red_pen(np_rgb)
    
    mask_no_green_pen = filter_green_pen(np_rgb)
    
    mask_no_blue_pen = filter_blue_pen(np_rgb)
    
    mask_gray_green_pens = mask_not_green & mask_not_gray & mask_right & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen

    if not tumor_region_jsonpath == None and Path(tumor_region_jsonpath).is_file():
        # check the tumor area annotation file
        region_border_list = parse_tumor_region_annotations(tumor_region_jsonpath)
        scaled_slide_dimensions = np_rgb.shape[:-1]
        # filter left only tumor or background
        if tumor_or_background == True:
            tumor_back_mask, _ = generate_tumor_mask(scaled_slide_dimensions, region_border_list)
        else:
            _, tumor_back_mask = generate_tumor_mask(scaled_slide_dimensions, region_border_list)
                    
        mask_all = mask_gray_green_pens & tumor_back_mask
    else:
        mask_all = mask_gray_green_pens
    
    mask_remove_small = filter_remove_small_objects(mask_all, min_size=500, output_type="bool", print_over_info=print_info)
    show_np_info = True if print_info == True else False
    rgb_remove_small = image_tools.mask_rgb(np_rgb, mask_remove_small, show_np_info=show_np_info)
    
    np_filtered_img = rgb_remove_small
    
    return np_filtered_img

def apply_image_filters_p62(np_img, tumor_region_jsonpath=None, tumor_or_background=True, print_info=True):
    """
    Apply filters to image as NumPy array and optionally save and/or display filtered images. For IHC-CD45 staining
    
    Args:
      np_img: Image as NumPy array.
      tumor_or_background: True: the filter only left tumor area; False: the filter only left background area
      
      @attention: removed
      <slide_num=None, info=None, save=False, display=False>
      
          slide_num: The slide number (used for saving/displaying).
          info: Dictionary of slide information (used for HTML display).
          save: If True, save image.
          display: If True, display image.
    
    Returns:
      Resulting filtered image as a NumPy array.
    """
    np_rgb = np_img
    
    if print_info == True:
        print('Filter noise of various colors and objects that are too small')
        
    mask_not_green = filter_green_channel(np_rgb, print_over_info=print_info)
    mask_not_gray = filter_grays(np_rgb, tolerance=14)
    # mask_right = filter_l_r_half(np_rgb, direction='right')
    mask_right = filter_leftmost_piece(np_rgb)
    
    mask_no_red_pen = filter_red_pen(np_rgb)
    
    mask_no_green_pen = filter_green_pen(np_rgb)
    
    mask_no_blue_pen = filter_blue_pen(np_rgb)
    
    mask_gray_green_pens = mask_not_green & mask_not_gray & mask_right & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen

    if not tumor_region_jsonpath == None and Path(tumor_region_jsonpath).is_file():
        # check the tumor area annotation file
        region_border_list = parse_tumor_region_annotations(tumor_region_jsonpath)
        scaled_slide_dimensions = np_rgb.shape[:-1]
        # filter left only tumor or background
        if tumor_or_background == True:
            tumor_back_mask, _ = generate_tumor_mask(scaled_slide_dimensions, region_border_list)
        else:
            _, tumor_back_mask = generate_tumor_mask(scaled_slide_dimensions, region_border_list)
                    
        mask_all = mask_gray_green_pens & tumor_back_mask
    else:
        mask_all = mask_gray_green_pens
    
    mask_remove_small = filter_remove_small_objects(mask_all, min_size=500, output_type="bool", print_over_info=print_info)
    show_np_info = True if print_info == True else False
    rgb_remove_small = image_tools.mask_rgb(np_rgb, mask_remove_small, show_np_info=show_np_info)
    
    np_filtered_img = rgb_remove_small
    
    return np_filtered_img

def apply_image_filters_psr(np_img, tumor_region_jsonpath=None, tumor_or_background=True, print_info=True):
    """
    Apply filters to image as NumPy array and optionally save and/or display filtered images. For IHC-PSR staining
    
    Args:
      np_img: Image as NumPy array.
      tumor_or_background: True: the filter only left tumor area; False: the filter only left background area
      
      @attention: removed
      <slide_num=None, info=None, save=False, display=False>
      
          slide_num: The slide number (used for saving/displaying).
          info: Dictionary of slide information (used for HTML display).
          save: If True, save image.
          display: If True, display image.
    
    Returns:
      Resulting filtered image as a NumPy array.
    """
    np_rgb = np_img
    
    if print_info == True:
        print('Filter noise of various colors and objects that are too small')
        
    grayscale = filter_rgb_to_grayscale(np_rgb)
    complement = filter_complement(grayscale)
    mask_hyst = filter_hysteresis_threshold(complement)
    
    mask_not_gray = filter_grays(np_rgb, tolerance=15)
    
    mask_no_red_pen = filter_red_pen(np_rgb)
    
    mask_no_green_pen = filter_green_pen(np_rgb)
    
    mask_no_blue_pen = filter_blue_pen(np_rgb)
    
    mask_gray_green_pens = mask_hyst & mask_not_gray & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen

    if not tumor_region_jsonpath == None and Path(tumor_region_jsonpath).is_file():
        # check the tumor area annotation file
        region_border_list = parse_tumor_region_annotations(tumor_region_jsonpath)
        scaled_slide_dimensions = np_rgb.shape[:-1]
        # filter left only tumor or background
        if tumor_or_background == True:
            tumor_back_mask, _ = generate_tumor_mask(scaled_slide_dimensions, region_border_list)
        else:
            _, tumor_back_mask = generate_tumor_mask(scaled_slide_dimensions, region_border_list)
                    
        mask_all = mask_gray_green_pens & tumor_back_mask
    else:
        mask_all = mask_gray_green_pens
    
    mask_remove_small = filter_remove_small_objects(mask_all, min_size=500, output_type="bool", print_over_info=print_info)
    show_np_info = True if print_info == True else False
    rgb_remove_small = image_tools.mask_rgb(np_rgb, mask_remove_small, show_np_info=show_np_info)
    
    np_filtered_img = rgb_remove_small
    
    return np_filtered_img


def apply_image_filters_he(np_img, tumor_region_jsonpath=None, tumor_or_background=True, print_info=True):
    """
    Apply filters to image as NumPy array and optionally save and/or display filtered images. For HE staining
    
    Args:
      np_img: Image as NumPy array.
      tumor_or_background: True: the filter only left tumor area; False: the filter only left background area
      
      @attention: removed
      <slide_num=None, info=None, save=False, display=False>
      
          slide_num: The slide number (used for saving/displaying).
          info: Dictionary of slide information (used for HTML display).
          save: If True, save image.
          display: If True, display image.
    
    Returns:
      Resulting filtered image as a NumPy array.
    """
    np_rgb = np_img
    
    if print_info == True:
        print('Filter noise of various colors and objects that are too small')
    mask_not_green = filter_green_channel(np_rgb, print_over_info=print_info) # this may only for H&E
#     rgb_not_green = image_tools.mask_rgb(np_rgb, mask_not_green)
#     save_display(save, display, info, rgb_not_green, slide_num, 2, "Not Green", "np_rgb-not-green")
    
    mask_not_gray = filter_grays(np_rgb, tolerance=12)
#     rgb_not_gray = image_tools.mask_rgb(np_rgb, mask_not_gray)
    
    mask_no_red_pen = filter_red_pen(np_rgb)
#     rgb_no_red_pen = image_tools.mask_rgb(np_rgb, mask_no_red_pen)
    
    mask_no_green_pen = filter_green_pen(np_rgb)
#     rgb_no_green_pen = image_tools.mask_rgb(np_rgb, mask_no_green_pen)
    
    mask_no_blue_pen = filter_blue_pen(np_rgb)
#     rgb_no_blue_pen = image_tools.mask_rgb(np_rgb, mask_no_blue_pen)
    
    mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
#     rgb_gray_green_pens = image_tools.mask_rgb(np_rgb, mask_gray_green_pens)

    if not tumor_region_jsonpath == None and Path(tumor_region_jsonpath).is_file():
        # check the tumor area annotation file
        
        region_border_list = parse_tumor_region_annotations(tumor_region_jsonpath)
        scaled_slide_dimensions = np_rgb.shape[:-1]
        # filter left only tumor or background
        if tumor_or_background == True:
            tumor_back_mask, _ = generate_tumor_mask(scaled_slide_dimensions, region_border_list)
        else:
            _, tumor_back_mask = generate_tumor_mask(scaled_slide_dimensions, region_border_list)
                    
        mask_all = mask_gray_green_pens & tumor_back_mask
    else:
        mask_all = mask_gray_green_pens
    
    mask_remove_small = filter_remove_small_objects(mask_all, min_size=500, output_type="bool", print_over_info=print_info)
    show_np_info = True if print_info == True else False
    rgb_remove_small = image_tools.mask_rgb(np_rgb, mask_remove_small, show_np_info=show_np_info)
    
    np_filtered_img = rgb_remove_small
    
    return np_filtered_img


def apply_image_roi_filters(np_img, tumor_region_jsonpath, print_info=True):
    """
    Apply filters to image as NumPy array after filtering left only ROI regions.
    
    except the mask regions, similar with function <apply_image_filters_he>
    Args:
      np_img: Image as NumPy array.
      
      @attention: removed
      <slide_num=None, info=None, save=False, display=False>
      
          slide_num: The slide number (used for saving/displaying).
          info: Dictionary of slide information (used for HTML display).
          save: If True, save image.
          display: If True, display image.
    
    Returns:
      Resulting filtered image as a NumPy array.
    """
    np_rgb = np_img
    
    if print_info == True:
        print('Filter non-ROI objects that are too small')
    
    region_border_list = parse_tumor_region_annotations(tumor_region_jsonpath)
    scaled_slide_dimensions = np_rgb.shape[:-1]
    tumor_mask, _ = generate_tumor_mask(scaled_slide_dimensions, region_border_list)
    
    mask_all = tumor_mask
    
    mask_remove_small = filter_remove_small_objects(mask_all, min_size=500, output_type="bool", print_over_info=print_info)
    show_np_info = True if print_info == True else False
    rgb_remove_small = image_tools.mask_rgb(np_rgb, mask_remove_small, show_np_info=show_np_info)
    
    np_filtered_img = rgb_remove_small
    
    return np_filtered_img
    

def parse_tumor_region_annotations(tumor_region_jsonpth):
    '''
    Args:
        tumor_region_jsonpth: example "TCGA_DATA_TUMOR_MASK_DIR/AIDA_annotation-<slide_id>.json"
    '''
    with open(tumor_region_jsonpth, 'r') as json_file:
        json_text = json_file.read()
        tumor_region_json = json.loads(json_text)
        
    '''
    tumor_region_json:
    {
        layers: [{
                    name: 
                    opacity: 
                    items: [items]
                },
                { }]
    }
    tumor_region_items:
    {
        bound:
        class:
        type:
        color:
        segments: [{point: {x:, y:}}]
        closed: true
    }
    '''   
    tumor_region_items = tumor_region_json['layers'][0]['items']
#     print(len(tumor_region_items))
    
    region_border_list = []
    point_number = []
    for item in tumor_region_items:
        segments = item['segments']
        vertex_xys = []
        for p in segments:
            point = p['point']
            vertex_xys.append( (point['x'] / ENV.SCALE_FACTOR, point['y'] / ENV.SCALE_FACTOR) )
        if len(vertex_xys) == 0:
            print('>>> skip an empty (0) region coordinate set! <<<')
            continue
        point_number.append(len(vertex_xys))
        region_border_list.append(vertex_xys)
    
    print('parsed tumor region from: {}, info: {} regions, with points number: {}'.format(tumor_region_jsonpth, str(len(region_border_list)), str(point_number)))
        
    return region_border_list

def generate_tumor_mask(scaled_slide_dimensions, region_border_list):
    '''
    Args:
        scaled_slide_dimensions: (large_w, large_h)
        region_border_list: [vertex_xys, vertex_xys, ...]
            vertex_xys: [(x, y), (x, y), ...]
    '''
    
    im = np.zeros(scaled_slide_dimensions, dtype="uint8")
    org_im = np.ones(scaled_slide_dimensions, dtype="uint8")
    
    for vertex_xys in region_border_list:
#         cv2.polylines(im, np.int32([vertex_xys]), 1, 1)
#         print(vertex_xys)
        cv2.fillPoly(im, np.int32([vertex_xys]), 1)
        
    tumor_mask_array = im
    non_tumor_mask_array = org_im - tumor_mask_array
    print('! apply tumor ROI mask with %d regions.' % (len(region_border_list)))
    
    return np.array(tumor_mask_array, dtype=bool), np.array(non_tumor_mask_array, dtype=bool)


if __name__ == '__main__':
    pass    
    
    
    
    
    
    
    
    
































