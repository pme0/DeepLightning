import matplotlib.pyplot as plt
import numpy as np
import librosa
from librosa.display import waveshow


def waveplot(
    path: str, 
    x_axis: str = "time", 
    save_plot: str = None, 
    show_plot: bool = True, 
    figsize=(8,4)
):
    """Display waveform in the time domain.

    Parameters
    ----------
    path : path to the audio file

    x_axis : type of x-axis (e.g. milliseconds, seconds, ...); 
        see `librosa.display.waveshow()` for all options

    save_plot : path to save the plot to; or `None` if
        the plot is not to be saved

    show_plot : whether to display plot

    """    

    # read audio file
    signal, sample_rate = librosa.load(path)

    # plot
    plt.figure(figsize=figsize)
    waveshow(signal, sr=sample_rate, x_axis=x_axis)
    plt.title("Audio waveplot")
    
    if save_plot is not None:
        plt.savefig(save_plot, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
