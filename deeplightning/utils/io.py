import os
from pytube import YouTube 


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

