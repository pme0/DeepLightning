import matplotlib.pyplot as plt
import numpy as np
import librosa
from librosa.display import specshow


def stft(path: str, n_fft: int, hop_length: int):
    """Display Short Term Fourier Transformation (STFT) in the time-frequency domain.

    Parameters
    ----------
    path : path to the audio file.
    
    n_fft: length of the windowed signal after padding with zeros.
    
    hop_length: number of audio samples between adjacent STFT columns.

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

     Parameters
    ----------
    path : path to the audio file.

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


def specplot(
    path: str,
    mode: str, 
    scale: str = None, 
    n_fft: int = 2048, 
    hop_length: int = 512, 
    figsize: tuple = (8,3), 
    save_plot: str = None,
    show_plot: bool = True,
    x_axis: str = "time",
):
    """Display Mel Frequency Cepstral Coefficients.

    Parameters
    ----------
    path : path to the audio file.
    mode : the type of features to be shown in the spectrogram. 
        Can be Short Time Fourier Transform (STFT) amplitude 
        (`stft_ampl`) or decibels (`stft_db`); Log Mel spectrogram 
        (logmel); Mel Frequency Cepstral Coefficients (`mfcc`)
    scale : the y-axis scale. Can be `linear` or `log`. For 
        `mode == "mfcc"` the scale is chosen automatically
    n_fft : 
    hop_length :
    figsize : the figure size
    x_axis : 
    """
    
    assert scale is None or scale in ["linear", "log"]

    signal, sample_rate = librosa.load(path)

    fig, ax = plt.subplots(figsize=figsize)

    if mode == "stft_ampl":
        stft_amplitude = np.abs(librosa.stft(signal, n_fft = n_fft))
        specshow(data = stft_amplitude, x_axis = x_axis, y_axis = scale, sr = sample_rate, hop_length=hop_length)
        frequency_type = 'STFT'
        colorbar_label = 'amplitude'
        scale_type = ', log'
    elif mode == "stft_db":
        stft_amplitude = np.abs(librosa.stft(signal, n_fft = n_fft))
        stft_decibel = librosa.amplitude_to_db(stft_amplitude)
        specshow(data = stft_decibel, x_axis = x_axis, y_axis = scale, sr = sample_rate, hop_length=hop_length)
        frequency_type = 'STFT'
        colorbar_label = 'decibel (dB)'
        scale_type = ', decibel'
    elif mode == "mfcc":
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        specshow(data = mfccs, x_axis = x_axis, y_axis = 'mel', sr = sample_rate, hop_length=hop_length)
        frequency_type = 'MFCC'
        colorbar_label = 'coefficients'  
        scale_type = ''
    elif mode == "logmel":
        S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        S = librosa.power_to_db(S, ref=np.max)
        specshow(data = S, x_axis = x_axis, y_axis = 'mel', sr = sample_rate, hop_length=hop_length)
        frequency_type = 'Log Mel'
        colorbar_label = 'decibel (dB)'  
        scale_type = ''
    else:
        raise NotImplementedError
        
    plt.xlabel("Time")
    plt.ylabel(f"Frequency (Hz)")
    plt.colorbar(label=colorbar_label, orientation="vertical", pad=0.02)
    plt.title("Spectrogram ({}{})".format(
        frequency_type if frequency_type is not None else '',
        scale_type if scale_type is not None else '',
    ))

    if save_plot is not None:
        plt.savefig(save_plot, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()