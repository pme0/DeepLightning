from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imagesize
from PIL import Image
import numpy as np

from deeplightning.utils.io_local import read_image


def plot_resized_image(
    image_fp: str,
    resize: Tuple[int,int] = None,
    display_image: bool = True,
    display_size: Tuple[int,int] = None,
    save_fp : str = None,
):
    """Plot resized image & saves with new exact pixel size.
    
    Parameters
    ----------
    image_fp : input image filepath
    resize : resize image to (width, height), in pixels; also used 
        to save the image with that exact pixel size (no borders)
    display_image : whether to display image
    display_size : size of displayed image, in inches (width, height)
    save_fp : output image save filepath
    """
    assert isinstance(resize, tuple) and len(resize) == 2
    
    image = Image.open(image_fp)
    image = image.convert('RGB')
    image = image.resize(resize)
    new_image = Image.new("RGB", resize, (255,255,255))
    position = ((resize[0] - image.width) // 2, (resize[1] - image.height) // 2)
    new_image.paste(image, position)
    
    if save_fp:
        new_image.save(save_fp)
    
    if display_image:
        fig = plt.figure(figsize=display_size)
        plt.imshow(new_image)
        plt.axis("off")
        plt.show()
        
    plt.close()