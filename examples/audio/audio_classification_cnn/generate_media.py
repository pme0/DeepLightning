import os
import sys
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ugly hack to be able to load modules as `import external.yolov5.ABC`.
# add path two levels upstream, to the main project folder.
current = os.path.dirname(os.path.realpath(__file__))
parent1 = os.path.dirname(current)
parent2 = os.path.dirname(parent1)
sys.path.append(parent2)
print(f"Added to system path: '{parent2}'")

from deeplightning.viz.audio.waveform import waveplot
from deeplightning.viz.audio.spectrum import spectrogram


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="~/data/fsd/", help="FSD dataset root folder")
    parser.add_argument("--sample_name", type=str, default="0_george_0.wav", help="FSD dataset root folder")
    args = parser.parse_args()
    return args


def main(args):
    
    # generate basic waveplot and spectrogram - nutcraker example
    
    audio_file = librosa.example('nutcracker')
    
    waveplot(
        path = audio_file, 
        x_axis = "time", 
        save_plot='examples/audio_classification/media/waveplot.png', 
        show_plot=False)

    spectrogram(
        path = audio_file,
        mode = "stft_db", 
        scale = "linear",
        save_plot='examples/audio_classification/media/spectrogram.png', 
        show_plot=False)


if __name__ == "__main__":

    args = parse_args()
    main(args)