def xcycwh2x0y0wh(box):
    """Convert bounding box format from `(xc,yc,w,h)` to 
    `(x0,y0,w,h)`. `xc` and `yc` are the bbox center coordinates, 
    `x0` and `y0` are the bbox corner coordinates. The 
    coordinates may be normalised or unnormalised.

    Args:
        box: bounding box array
    """
    return [box[0] - box[2] / 2, box[1] - box[3] / 2, box[2], box[3]]


def xcycwhn2x0y0wh(box, width: int, height: int):
    """Convert bounding box format from normalised `(xc,yc,w,h)` 
    to unnormalised `(x0,y0,w,h)`. `xc` and `yc` are the bbox 
    center coordinates, `x0` and `y0` are the bbox corner coordinates.

    Args:
        box: bounding box array
        width: image width
        height: image height
    """
    return [width * (box[0] - box[2] / 2), height * (box[1] - box[3] / 2), width * box[2], height * box[3]]


def xcycwhn2xcycwh(box, width: int, height: int):
    """Convert bounding box format from normalised `(xc,yc,w,h)` 
    to unnormalised format `(xc,yc,w,h)`. `xc` and `yc` are the bbox 
    center coordinates.

    Args:
        box: bounding box array
        width: image width
        height: image height
    """
    return [width * box[0], height * box[1], width * box[2], height * box[3]]


def x0y0x1y1_to_x0y0wh(box):
    """Convert bounding box format from unnormalised `(x0,y0,x1,y1)` 
    to unnormalised format `(x0,y0,w,h)`. `x0` and `y0` are the bbox 
    top-left corner coordinates and `x1` and `y1` are the bottom-right
    bbox corner coordinates.

    Args:
        box: bounding box array
    """
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]