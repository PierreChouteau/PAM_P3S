from os import listdir, mkdir
from os.path import isfile, join, isdir
from librosa import load
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
from multiprocessing import Pool
from functools import partial


def extract_wav_instru(
    all_mics: list,
    path_prefix: str,
):
    """Extract all 8 wav files contained in the folder designed by the path_prefix."""

    # Get all the files in the folder
    raw_files = listdir(path_prefix)

    wavefiles = []
    # Reorder the files
    for name in all_mics:
        for file in raw_files:
            if isfile(join(path_prefix, file)) and (
                name == file.rstrip(".wav").split(" ")[2]
            ):
                data, sr = load(join(path_prefix, file))
                wavefiles.append(data)
    array_wavefiles = np.zeros((len(wavefiles), *wavefiles[0].shape))
    for k in range(len(wavefiles)):
        array_wavefiles[k] = wavefiles[k]
    return array_wavefiles, sr


def extract_wav_instru_multiprocess(
    all_mics: list,
    path_prefix: str,
):
    # Get all the files in the folder
    raw_files = listdir(path_prefix)

    wavefile_names = []
    # Reorder the files
    for name in all_mics:
        for file in raw_files:
            if isfile(join(path_prefix, file)) and (
                name == file.rstrip(".wav").split(" ")[2]
            ):
                wavefile_names.append(join(path_prefix, file))
    kwargs = {"sr": None}
    with Pool() as p:
        data = p.map(partial(load, **kwargs), wavefile_names)
    wave_data = np.zeros((len(wavefile_names), *data[0][0].shape))
    for i, (wav, sr) in enumerate(data):
        wave_data[i] = wav
    return wave_data, sr


def split_scale(
    wavefile: np.ndarray,
    start_time: float,
    nb_notes: int,
    bpm: float = 60,
    sr: int = 48000,
):
    """Split the individual notes from the scale played in the wav file."""

    # Compute the number of samples per beat
    window_width = int(60 * sr / bpm)

    # Start and stop sample
    start_sample = int(start_time * sr)

    # Initialize the array
    notes = np.zeros((nb_notes, window_width))

    time_vec = np.linspace(0, len(wavefile) / sr, len(wavefile))

    plt.subplot(221)
    plt.plot(
        time_vec[: start_sample + window_width + sr],
        wavefile[: start_sample + window_width + sr],
    )
    plt.vlines(time_vec[start_sample], -1, 1, color="red")
    plt.vlines(time_vec[start_sample + window_width], -1, 1, color="green")

    plt.subplot(222)
    plt.plot(
        time_vec[start_sample + (nb_notes - 1) * window_width :],
        wavefile[start_sample + (nb_notes - 1) * window_width :],
    )
    plt.vlines(
        time_vec[start_sample + (nb_notes - 1) * window_width], -1, 1, color="red"
    )
    plt.vlines(time_vec[start_sample + nb_notes * window_width], -1, 1, color="green")

    plt.subplot(212)
    plt.plot(time_vec, wavefile)
    plt.vlines(time_vec[start_sample], -1, 1, color="red")

    # Extract the notes
    for k in range(nb_notes):
        start = start_sample + k * window_width
        stop = start_sample + (k + 1) * window_width
        notes[k] = wavefile[start:stop]
        plt.vlines(time_vec[stop], -1, 1, color="green")
    plt.show()
    return notes


def save_notes(
    notes: np.ndarray, instrument: str, pitch_midi_start: int, sr: int, path: str
):
    pitch_midi = pitch_midi_start
    for note in notes:
        pitch_midi += 1
        name = f"{pitch_midi}_{instrument}.wav"
        sf.write(join(path, name), note, samplerate=sr)


def save_notes_multiprocess(
    notes: np.ndarray, instrument: str, pitch_midi_start: int, sr: int, path: str
):
    names = [
        f"{pitch_midi}_{instrument}.wav"
        for pitch_midi in range(pitch_midi_start, pitch_midi_start + len(notes))
    ]
    paths = [join(path, name) for name in names]
    with Pool() as p:
        p.starmap(sf.write, zip(paths, notes, [sr] * len(notes)))


def extract_and_save_notes(
    path: str, instrument: str, params_instru: tuple, pizz: bool = False
):
    mic_names = ["L", "C", "R", "Violin1", "Violin2", "Flute", "Clarinet", "Cello"]
    start_time, nb_notes, bpm, midi_pitch = params_instru
    total_path = join(path, instrument)
    if pizz:
        total_path = join(total_path, "Pizz")
    wavefiles, sr = extract_wav_instru_multiprocess(mic_names, total_path)
    index_instru = mic_names.index(instrument)
    notes = split_scale(wavefiles[index_instru], start_time, nb_notes, bpm, sr)
    response = input("Save the notes? (y/n)")
    if response != "y":
        return
    save_path = join(total_path, "Notes")
    if not isdir(save_path):
        mkdir(save_path)
    save_notes_multiprocess(notes, instrument, midi_pitch, sr, save_path)


def main():
    path_scales = "Sounds/Scales/"

    param_cello = (0.7, 32, 65.5, 20)

    extract_and_save_notes(path_scales, "Cello", param_cello, pizz=False)


if __name__ == "__main__":
    main()
