import matplotlib.pyplot as plt
import numpy as np
import librosa
from librosa.display import waveshow


def waveform(path: str):
    """Display Waveform in the time domain.

    Parameters
    ----------
    path : path to the audio file.

    """    

    signal, sample_rate = librosa.load(path)
    waveshow(signal, sr=sample_rate)
