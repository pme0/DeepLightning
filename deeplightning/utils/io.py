from typing import Any, Union, Tuple, Optional, List
import os
from pytube import YouTube 
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pickle
import json



def download_video_from_youtube(url: str, savedir: str, savename: str):
    """Download video from YouTube.
    
    To trim the video use the command
    ```
        ffmpeg -i input.mp4 -ss 00:00:00 -t 00:00:05 output.mp4
    ```
    where `ss` is the start and `t` is the duration; in the 
    example, extract the first 5 seconds of the video.

    Parameters
    ----------
    url : the YouTube URL of the video to be downloaded
    savedir : the directory that the video is saved to
    savename : the filename that the video is saved to
    
    """
      
    try:
        yt = YouTube(url) 
    except: 
        print(f"Connection Error (url='{url}')")
        
    saved_to = yt.streams.filter(
        progressive = True, 
        file_extension = "mp4").first().download(
            output_path = savedir, 
            filename = f"{savename}.mp4")
    
    print(f"Downloaded '{url}' to '{saved_to}'") 


def read_video(
    video_path: str,
) -> Tuple[np.ndarray, str, int]:
    """Read video from file into a numpy array.
    
    Parameters
    ----------
    video_path: path to video.
    
    Returns
    -------
    tuple `(video, name, frames)` where `video` is the video array; 
        `name` is the original video name; `frames` is the number of frames.
    """
    assert isinstance(video_path, str), "Video path must be a string type"
    
    # get video filename (excluding extension)
    video_name = video_path.split(".")
    if len(video_name) > 2:
        raise ValueError("Invalid filename: video contains a dot (.) in its filename besides the one for the type extension")
    video_name = video_name[0].split("/")[-1]

    recording = cv2.VideoCapture(video_path)
    recording_array = []
    if recording.isOpened() == False:
        print("Error while opening the video")
    while recording.isOpened():
        ret, frame = recording.read()
        # ret is a boolean variable telling if the video is read correctly or not
        if ret == True:
            recording_array.append(frame)
        else:
            break
    recording.release()
    recording_array = np.array(recording_array)
    return recording_array, video_name, recording_array.shape[0]


def save_video_frames_as_images(
    video: np.ndarray, 
    write_path: str,
    write_name: str, 
) -> None:
    """
    """
    os.makedirs(write_path, exist_ok=True)
    for i in range(np.shape(video)[0]):
        frame = video[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        filename = os.path.join(write_path, write_name+'-'+'{0:08}'.format(i)+'.png')
        plt.imsave(filename, frame)


def write_video_frames_to_images(
    video_path: str,
    write_path: str,
    write_name: str,
) -> None:
    """Read video and write video frames to image files.
    Args:
        :video_path: path to video to be read.
        :write_path: path to write frames to.
        :write_name: prefix name for the saved images.
    """
    video_array, _, _ = read_video(video_path)
    save_video_frames_as_images(video=video_array, write_path=write_path, write_name=write_name)


def read_image(
    image_path: str,
    device: torch.device,
    resize: int = None,
    transform: str = None,
) -> torch.Tensor:
    """Read image from file into torch tensor.
    Args:
        :image_path:
        :device:
        :resize:
        :transform:
    Returns:
        :torch.Tensor: the image tensor
    """

    image = Image.open(image_path)
    image = image.convert('RGB')
    if resize:
        assert isinstance(resize, int)
        image = T.Resize((resize, resize))(image)
    if transform:
        image = transform(image)
    image = T.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image


def pickle_obj(obj: Any, path: str):
    assert path.endswith(".pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def unpickle_obj(path: str):
    assert path.endswith(".pkl")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def write_dict_as_json(obj: dict, path: str, indent: int = None):
    assert path.endswith(".json")
    assert isinstance(obj, dict)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
