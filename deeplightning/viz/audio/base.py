import librosa
import matplotlib.pyplot as plt
import numpy as np


def waveform(path: str):
    """Display Waveform in the time domain.

    Args
    ----------
    :path: path to the audio file

    """    

    signal, sample_rate = librosa.load(path)
    librosa.display.waveshow(signal, sr=sample_rate)



def stft(path: str, n_fft: int, hop_length: int):
    """Display Short Term Fourier Transformation(STFT) in the time-frequency domain.

    Args
    ----------
    :path: path to the audio file
    :n_fft: length of the windowed signal after padding with zeros
    :hop_length: number of audio samples between adjacent STFT columns

    """    
    # load data
    signal, sample_rate = librosa.load(path)

    # compute spectrogram
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # compute spectrogram
    spectrogram = np.abs(stft)

    # plot
    plt.figure(figsize=(3,5))
    librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.title("Short Term Fourier Transformation-STFT")



def power_spectrum(path: str):
    """Display Power Spectrum.

     Args
    ----------
    :path: path to the audio file

    """

    # load data
    signal, sample_rate = librosa.load(path)

    # calcultate spectrum
    fft = np.fft.fft(signal)  # fast fourier transform
    spectrum = np.abs(fft)    # magnitude
    
    # create frequency
    f = np.linspace(0, sample_rate, len(spectrum))
    half_spectrum = spectrum[:int(len(spectrum)/2)]  # take half of the spectrum and frequency as it is a mirror image
    half_f = f[:int(len(spectrum)/2)]

    # plot
    plt.figure(figsize=(5,2))
    plt.plot(half_f, half_spectrum, alpha=0.4)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum")


def mfcc(path: str):
    """Display Mel Frequency Cepstral Coefficients.

    Args
    ----------
    :path: path to the audio file

    """

