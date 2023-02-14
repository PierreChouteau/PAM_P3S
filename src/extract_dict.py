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
                name
                == file.rstrip(".wav").split(" ")[2]  # For non pizz
                # name
                # == file.rstrip(".wav").split(" ")[3]
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

    if len(wavefile) < start_sample + nb_notes * window_width:
        tmp = np.zeros(start_sample + nb_notes * window_width + 1)
        tmp[: len(wavefile)] = wavefile
        wavefile = tmp

    time_vec = np.linspace(0, len(wavefile) / sr, len(wavefile))

    plt.subplot(212)
    plt.plot(time_vec, wavefile)
    plt.vlines(time_vec[start_sample], -1, 1, color="red")

    # Extract the notes
    for k in range(nb_notes):
        start = start_sample + k * window_width
        stop = start_sample + (k + 1) * window_width
        notes[k] = wavefile[start:stop]
        plt.vlines(time_vec[stop], -1, 1, color="green")
    plt.subplot(221)
    plt.plot(
        time_vec[: start_sample + window_width + sr],
        wavefile[: start_sample + window_width + sr],
    )
    plt.vlines(time_vec[start_sample], -1, 1, color="red")
    plt.vlines(time_vec[start_sample + window_width], -1, 1, color="green")

    plt.subplot(222)
    plt.plot(
        time_vec[start:],
        wavefile[start:],
    )
    plt.vlines(time_vec[start], -1, 1, color="red")
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


def compute_spectro(data: np.ndarray, nfft: int):
    return np.square(np.abs(np.fft.rfft(data, nfft)))


def save_matrix(instrument: str, matrix: np.ndarray, path: str):
    save_path = join(path, "Matrix")
    if not isdir(save_path):
        mkdir(save_path)
    filename = join(save_path, f"Matrix_{instrument}.npy")
    np.save(filename, matrix)
    load_matrix(filename)


def create_spectral_matrix(
    instrument: str, path: str, n_notes: int, pizz: bool = False, nfft: int = 4096
):
    # n_notes is the max number of notes among the instruments.
    # If the current instrument has less notes, the matrix will be padded with zeros.

    path_notes = join(path, instrument, "Notes")
    files = [join(path_notes, file) for file in listdir(path_notes)]
    files.sort()

    if pizz:
        path_pizz = join(path, instrument, "Pizz", "Notes")
        files_pizz = [join(path_pizz, file) for file in listdir(path_pizz)]
        files_pizz.sort()
        files += files_pizz

    spectral_matrix = np.zeros((nfft // 2 + 1, n_notes))
    for i in range(len(files)):
        file = files[i]
        data, sr = sf.read(file)
        spectral_matrix[:, i] = compute_spectro(data, nfft)
    for i in range(n_notes - len(files)):
        spectral_matrix[:, len(files) + i] = spectral_matrix[:, i]
        spectral_matrix[: -i - 1, len(files) + i] = 0
    save_matrix(instrument, spectral_matrix, path)


def load_matrix(filename: str):
    mat = np.load(filename)
    plt.imshow(np.log(mat + 1e-6), aspect="auto", origin="lower")
    plt.show()
    return mat


def modify_wav(filename: str, threshold_ratio: int, show: bool = False):
    data, sr = sf.read(filename)
    power = np.square(data)
    threshold = threshold_ratio * np.max(power)
    for i in range(len(power)):
        if power[i] > threshold:
            start = i
            break
    if show:
        plt.plot(data)
        plt.plot(np.arange(len(data))[start:], data[start:])
        plt.show()
    sf.write(filename, data[start:], sr)


def main():
    path_scales = join("Sounds", "Scales")

    # params_instru = (start_time, nb_notes, bpm, midi_pitch_start)
    param_violin1 = (0.73, 39, 30.3, 55)
    param_violin2 = (1.1, 24, 30.75, 57)
    param_flute = (0, 36, 29.3, 60)
    param_clarinet = (0.7, 40, 26, 50)
    param_cello = (1.4, 32, 30, 36)

    param_violin1_pizz = (0.73, 39, 30.3, 55)
    param_violin2_pizz = (0, 24, 30.75, 57)
    param_cello_pizz = (1, 32, 30, 36)

    # extract_and_save_notes(path_scales, "Violin1", param_violin1, pizz=False)
    # extract_and_save_notes(path_scales, "Violin2", param_violin2, pizz=False)
    # extract_and_save_notes(path_scales, "Flute", param_flute, pizz=False)
    # extract_and_save_notes(path_scales, "Clarinet", param_clarinet, pizz=False)
    # extract_and_save_notes(path_scales, "Cello", param_cello, pizz=False)

    # extract_and_save_notes(path_scales, "Violin1", param_violin1_pizz, pizz=True)
    # extract_and_save_notes(path_scales, "Violin2", param_violin2_pizz, pizz=True)
    # extract_and_save_notes(path_scales, "Cello", param_cello_pizz, pizz=True)

    # create_spectral_matrix("Violin1", path_scales, n_notes=78, pizz=True)
    # create_spectral_matrix("Violin2", path_scales, n_notes=78, pizz=True)
    # create_spectral_matrix("Flute", path_scales, n_notes=78, pizz=False)
    # create_spectral_matrix("Clarinet", path_scales, n_notes=78, pizz=False)
    # create_spectral_matrix("Cello", path_scales, n_notes=78, pizz=True)

    path_notes_violin1 = join(path_scales, "Violin1", "Notes")
    path_notes_violin1_pizz = join(path_scales, "Violin1", "Pizz", "Notes")
    path_notes_violin2 = join(path_scales, "Violin2", "Notes")
    path_notes_violin2_pizz = join(path_scales, "Violin2", "Pizz", "Notes")
    path_notes_flute = join(path_scales, "Flute", "Notes")
    path_notes_clarinet = join(path_scales, "Clarinet", "Notes")
    path_notes_cello = join(path_scales, "Cello", "Notes")

    files_violin1 = [
        join(path_notes_violin1, file) for file in listdir(path_notes_violin1)
    ]
    files_violin1_pizz = [
        join(path_notes_violin1_pizz, file) for file in listdir(path_notes_violin1_pizz)
    ]
    files_violin2 = [
        join(path_notes_violin2, file) for file in listdir(path_notes_violin2)
    ]
    files_violin2_pizz = [
        join(path_notes_violin2_pizz, file) for file in listdir(path_notes_violin2_pizz)
    ]
    files_flute = [join(path_notes_flute, file) for file in listdir(path_notes_flute)]
    files_clarinet = [
        join(path_notes_clarinet, file) for file in listdir(path_notes_clarinet)
    ]
    files_cello = [join(path_notes_cello, file) for file in listdir(path_notes_cello)]

    all_notes_files = (
        files_violin1
        + files_violin1_pizz
        + files_violin2
        + files_violin2_pizz
        + files_flute
        + files_clarinet
        + files_cello
    )
    for file in all_notes_files:
        modify_wav(file, 5e-3)


if __name__ == "__main__":
    main()
