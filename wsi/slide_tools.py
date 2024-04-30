'''

@author: yang hu
'''
import math
import sys

from PIL import Image
from openslide import open_slide

from support.env import ENV
from wsi import image_tools


Image.MAX_IMAGE_PIXELS = None
sys.path.append("..")


def just_get_slide_from_openslide(slide_filepath):
    '''
    very directly get slide from openslide openable files
    '''
    slide = open_slide(slide_filepath)
    return slide
    
def just_get_slide_from_normal(slide_filepath):
    '''
    very directly get slide from normal image files, like png or jpg
    '''
    img = Image.open(slide_filepath)
    return img


def original_slide_and_scaled_pil_image(slide_filepath, scale_factor=ENV.SCALE_FACTOR, print_opening=False):
    """
    Convert a WSI training slide to a scaled-down PIL image.
    
    Also return the slide object
    """
    if print_opening == True:
        print("Opening Slide: %s" % slide_filepath)
    slide = open_slide(slide_filepath)
    
    # get the shape of original WSI slide (.svs .tif)
    """ ! width and height are inverted as in PyTorch """ 
    large_w, large_h = slide.dimensions
    # set the target shape of the scaled small PIL image (.png)
    small_w = math.floor(large_w / scale_factor)
    small_h = math.floor(large_h / scale_factor)
    # automatically get the level of scaled small slide 
    level = slide.get_best_level_for_downsample(scale_factor)
    # get the PIL image for original WSI slide and transfer it into RGB format
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    # scale the WSI image (with the reshape function)
    img = whole_slide_image.resize((small_w, small_h), Image.BILINEAR)
    
    return img, slide

def slide_to_scaled_pil_image(slide_filepath, scale_factor=ENV.SCALE_FACTOR, print_opening=True):
    """
    Convert a WSI training slide to a scaled-down PIL image.
    *reference the code from: https://github.com/deroneriksson/python-wsi-preprocessing
    
    Args:
      slide_filepath: File path of the original WSI slide.
    
    Returns:
      Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    if print_opening == True:
        print("Opening Slide: %s" % slide_filepath)
    slide = open_slide(slide_filepath)
    
    # get the shape of original WSI slide (.svs .tif)
    """ ! width and height are inverted as in PyTorch """ 
    large_w, large_h = slide.dimensions
    # set the target shape of the scaled small PIL image (.png)
    small_w = math.floor(large_w / scale_factor)
    small_h = math.floor(large_h / scale_factor)
    # automatically get the level of scaled small slide 
    level = slide.get_best_level_for_downsample(scale_factor)
    # get the PIL image for original WSI slide and transfer it into RGB format
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    # scale the WSI image (with the reshape function)
    img = whole_slide_image.resize((small_w, small_h), Image.BILINEAR)
    
    return img, large_w, large_h, small_w, small_h

def slide_to_scaled_np_image(slide_filepath, scale_factor=ENV.SCALE_FACTOR, print_opening=True):
    """
    Convert a WSI training slide to a scaled-down NumPy image.
    *reference the code from: https://github.com/deroneriksson/python-wsi-preprocessing
    
    Args:
      slide_filepath: File path of the original WSI slide.
    
    Returns:
      Tuple consisting of scaled-down NumPy image, original width, original height, new width, and new height.
    """
    pil_img, large_w, large_h, small_w, small_h = slide_to_scaled_pil_image(slide_filepath, scale_factor,
                                                                            print_opening)
    np_img = image_tools.pil_to_np_rgb(pil_img)
    
    return np_img, large_w, large_h, small_w, small_h


def small_to_large_mapping(small_pixel):
    """
    Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.
    
    Args:
      small_pixel: The scaled-down width and height.
      large_dimensions: The width and height of the original whole-slide image.
    
    Returns:
      Tuple consisting of the scaled-up width and height.
    """
    small_x, small_y = small_pixel
    large_x = math.floor(ENV.SCALE_FACTOR * small_x)
    large_y = math.floor(ENV.SCALE_FACTOR * small_y)
    
    return large_x, large_y


if __name__ == '__main__':
    pass