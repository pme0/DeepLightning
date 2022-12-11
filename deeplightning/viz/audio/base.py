import matplotlib.pyplot as plt
import numpy as np
import librosa
from librosa.display import waveshow, specshow


def waveform(path: str):
    """Display Waveform in the time domain.

    Args
    ----------
    :path: path to the audio file

    """    

    signal, sample_rate = librosa.load(path)
    waveshow(signal, sr=sample_rate)



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
    specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
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


def spectrogram(
    path: str,
    mode: str, 
    scale: str = None, 
    n_fft: int = 2048, 
    hop_length: int = 512, 
    figsize: tuple = (5,5), 
):
    """Display Mel Frequency Cepstral Coefficients.

    Args
    ----------
    :path: path to the audio file
    :mode: 
    :scale: 
    :n_fft: 
    :hop_length:
    :figsize:

    """
    
    assert scale is None or scale in ["linear", "log", "mel"]

    signal, sample_rate = librosa.load(path)

    fig, ax = plt.subplots(figsize=figsize)

    if mode == "stft_ampl":
        stft_amplitude = np.abs(librosa.stft(signal, n_fft = n_fft))
        specshow(data = stft_amplitude, x_axis = 'time', y_axis = scale, sr = sample_rate, hop_length=hop_length)
        frequency_type = 'STFT'
        colorbar_label = 'amplitude'
        scale_type = ', log'
    elif mode == "stft_db":
        stft_amplitude = np.abs(librosa.stft(signal, n_fft = n_fft))
        stft_decibel = librosa.amplitude_to_db(stft_amplitude)
        specshow(data = stft_decibel, x_axis = 'time', y_axis = scale, sr = sample_rate, hop_length=hop_length)
        frequency_type = 'STFT'
        colorbar_label = 'decibel (dB)'
        scale_type = ', decibel'
    elif mode == "mfcc":
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        specshow(data = mfccs, x_axis = 'time', y_axis = 'mel', sr = sample_rate, hop_length=hop_length)
        frequency_type = 'MFCC'
        colorbar_label = 'coefficients'  
        scale_type = ''
    else:
        raise NotImplementedError
        
    plt.xlabel("Time")
    plt.ylabel(f"Frequency (Hz)")
    plt.colorbar(label=colorbar_label, orientation="horizontal")
    plt.title("Spectrogram ({}{})".format(
        frequency_type if frequency_type is not None else '',
        scale_type if scale_type is not None else '',
    ))