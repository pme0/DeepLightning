from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import imagesize
import PIL


def convert_image_sequence_to_video(image_folder):
    """
    the ffmpeg command is
    ```
    ffmpeg -pattern_type glob -i 'img_*.png' -c:v libx264 -vf fps=30 -pix_fmt yuv420p output.mp4
    ```

    ffmpeg reads images alphabetically, so ensure that filenames 
    containing frame numbers ('img_0', 'img_1', ...) are left-padded 
    with zeros ('img_0000', 'img_0001', ...); otherwise images will 
    not appear in the correct order.
    """
    pass


def plot_image_and_bboxes(
    image_path: str, 
    bboxes: List[Dict] = [],
    resize: int = None, 
    save_path: str = None, 
    show_image: bool = True
):
    """Plot an image together with a set of bounding boxes and class labels

    Example:
    ```
    from deeplightning.viz.image.bboxes import plot_image_and_bboxes
    img_path = "media/eye.jpg"
    bboxes = [
        {"class": "iris",  "box": [253, 245, 244, 240], "format": "xcycwh"},
        {"class": "pupil", "box": [244, 243, 68+x, 64], "format": "xcycwh"}]
    plot_image_and_bboxes(image_path=img_path, bboxes=bboxes, resize=500, 
                    save_path=None, show_image=True)
    ```
    """
    
    image = PIL.Image.open(image_path)
    img_w, img_h = imagesize.get(image_path)
    assert img_w == img_h  # assumes square image
    if resize is None:
        resize = img_w
    else:
        image = image.resize((resize, resize))
        
    # params
    font_factor = 0.03
    label_width_factor = 0.025
    label_heigth_factor = 0.05
    
    # plot
    # figure size is set in pixels;
    # be default it seems 30 pixels in each direction are allocated 
    # to white border/axes, so adjustment for that is required if
    # the saved image is to be exactly `resize x resize`;
    colors = sns.color_palette("Set2")
    px = 1/plt.rcParams['figure.dpi']
    resize_extra = 30 + resize
    fig, ax = plt.subplots(1,1, figsize=(resize_extra*px, resize_extra*px))
    plt.imshow(image)
    for i, bbox in enumerate(bboxes):
        
        if bbox["format"] == "xcycwh":
            box = {
                "x_LowerLeft": bbox["box"][0] - bbox["box"][2]/2,
                "y_LowerLeft": bbox["box"][1] - bbox["box"][3]/2,
                "width": bbox["box"][2],
                "height": bbox["box"][3],
            }
            if resize is not None:
                box["x_LowerLeft"] = box["x_LowerLeft"] * resize / img_w
                box["y_LowerLeft"] = box["y_LowerLeft"] * resize / img_h
                box["width"] = box["width"] * resize / img_w
                box["height"] = box["height"] * resize / img_h
        
        # boundings box
        ax.add_patch(
            patches.Rectangle(
                xy=(box["x_LowerLeft"], box["y_LowerLeft"]), 
                width=box["width"], 
                height=box["height"], 
                color=colors[i],
                alpha=0.4,
            )
        )
        label_conf = None if "conf" not in bbox else bbox["conf"]
        label_text = "{}{}{:.2f}".format(bbox["class"].lower(), "" if label_conf is None else " ", round(label_conf,2))
        # label box
        label_width = label_width_factor * resize_extra * len(label_text)
        label_heigth = label_heigth_factor * resize_extra
        ax.add_patch(
            patches.Rectangle(
                xy=(box["x_LowerLeft"], box["y_LowerLeft"] - label_heigth), 
                width=label_width, 
                height=label_heigth, 
                facecolor=colors[i],
                edgecolor='none',
                alpha=0.8,
            )
        )
        # label text
        plt.annotate(
            text=label_text, 
            xy=(box["x_LowerLeft"], box["y_LowerLeft"]),
            color='white',
            ha="left",
            va = "bottom",
            alpha=1,
            font=dict(size=int(font_factor * resize_extra)),
        )
    # https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image
    plt.gca().set_axis_off()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show_image:
        plt.show()
    plt.close()

    
