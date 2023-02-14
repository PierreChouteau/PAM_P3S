# import libraries
import IPython.display as ipd
from IPython.display import display

import os
import pyroomacoustics as pra
import stempeg
import librosa
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


def get_files(path_in, extension=".mp4"):

    """
    This function take the path to a folder as input
    and creates a list of the files contained inside and their names


    Inputs:
    ------------------------------------------------------
    path_in: string
        path to the folder containing the files to be processed

    extension: string
        extension of the files to be processed (default = '.mp4')

    Output:
    ------------------------------------------------------
    files_in: list of string (n_files)
        list of the files contained in the folder

    files_title: list of string (n_files)
        list of the names of the files contained in the folder

    """
    files_in = []
    files_title = []

    # extract files
    for r, _, f in os.walk(path_in):
        for file in f:
            if extension in file:
                # file address
                files_in.append(os.path.join(r, file))
                # file author + song
                files_title.append(file[:-len(extension)])

    files_in.sort()
    files_title.sort()

    return files_in, files_title


def room_sources_micro(
    audio_list,
    rate,
    start_time,
    audio_length,
    room,
    source_locations,
    microphone_locations,
    microphone_names,
    source_dir=None,
    mic_dir=None,
    display_room=False,
):

    """

    this function process a song from MUSDB18 dataset
    as a recording in a shoebox room (defined with its
    geometry, room absorption, signal locations and
    microphones locations) into a multichannel STFT.

    Inputs:
    ------------------------------------------------------
    audio_list: list or numpy.ndarray (n_sources, n_channels, n_samples)
        list of the audio files to be processed

    rate: int
        sampling rate

    audio_length: int
        length of the audio signal

    room: room object (pyroomacoustics) - see pyroomacoustics documentation
        WARNING: the definition of the room is done outside the function, and should
        not contain any source or microphone. Just the geometry, the absorption and the sampling rate.

    source_locations: list of list (n_sources, n_dimensions)
        localization of the sources in the room

    microphone_locations: list of string (n_channels)
        localization of the micros in the room (warning, locs in `np.c_` class)

    microphone_names: list of string (n_channels)
        name of the microphones

    source_dir:
        source directivity (pyroomacoustics directivity object)

    mic_dir:
        microphone directivity (pyroomacoustics directivity object)

    display_room:
        bool (display room geometry)



    Output
    ------------------------------------------------------

    room: room object (pyroomacoustics)
        room object with the sources and microphones added

    separate_recordings:
        recordings of the sources in the room (n_sources, n_channels, n_samples)

    mics_signals:
        recordings of the microphones in the room (n_channels, n_samples)

    """

    # Add sources
    for source, source_loc in zip(audio_list, source_locations):
        signal_channel = librosa.core.to_mono(source[:, start_time*rate: (start_time + audio_length) * rate])
        room.add_source(
            position=source_loc, directivity=source_dir, signal=signal_channel
        )

    # Add microphone array
    mic_array = pra.MicrophoneArray(microphone_locations, rate)
    room.add_microphone_array(mic_array, directivity=mic_dir)

    if display_room:
        fig, ax = room.plot()
        lim = np.max(room.shoebox_dim)
        ax.set_xlim([0, lim])
        ax.set_ylim([0, lim])
        ax.set_zlim([0, lim])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Recordings
    separate_recordings = room.simulate(return_premix=True)
    mics_signals = np.sum(separate_recordings, axis=0)

    return room, separate_recordings, mics_signals


def spectrogram_from_mics_signal(
    mics_signals,
    microphone_names,
    rate=44100,
    L=2048,
    hop=512,
    type="stft",
    display_audio=False,
    display_spectrogram=False,
):

    """
    This function process an audio signal with multiple channels
    and return the STFT multi-channel of the signal


    Inputs:
    ------------------------------------------------------
    mics_signal: numpy.ndarray (n_channels, n_samples)
        audio signal from each microphone

    microphone_names: list of string (n_channels)
        name of the microphones

    rate: int
        sampling rate (44100 default)

    L: int
        frame size (2048 default)

    hop: int
        hop length (512 default)

    type: string (default 'stft')
        define the type of the transformation

    display_audio:
        bool (display audio signal)

    display_spectrogram:
        bool (display spectrogram)


    Output:
    ------------------------------------------------------
    X: (n_frames, n_frequencies, n_channels)
        multi-channel STFT of mics_signal

    """

    if type == "stft":
        # STFT parameters
        win_a = pra.hamming(L)
        # win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

        # Observation vector in the STFT domain
        X = pra.transform.stft.analysis(mics_signals.T, L, hop, win=win_a)

    elif type == "cqt":
        X = librosa.cqt(
            mics_signals,
            sr=rate,
            hop_length=hop,
            fmin=None,
            n_bins=84,
            bins_per_octave=12,
            tuning=0.0,
            filter_scale=1,
            norm=1,
            sparsity=0.01,
            window="hann",
            scale=True,
            pad_mode="constant",
            res_type=None,
            dtype=None,
        )
        X = X.transpose(2, 1, 0)

    else:
        return NameError

    if display_audio:
        for microphone_n in range(len(microphone_names)):
            print(f"Microphone {microphone_names[microphone_n]}")
            display(ipd.Audio(mics_signals[microphone_n], rate=rate))

    if display_spectrogram:
        for microphone_n in range(len(microphone_names)):
            plt.figure(figsize=(10, 4))
            plt.specgram(mics_signals[microphone_n], NFFT=L, Fs=rate)
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.title(f"Spectrogram of microphone {microphone_names[microphone_n]}")
            plt.show()

    return X


def shoebox_room(
    room_dimension,
    abs_coef=0.35,
    rate=44100,
    max_order=15,
    sigma2_awgn=1e-8,
    display_room=False,
):

    """
    This function creates a shoebox room (defined
    with its geometry, room absorption, source
    locations and microphones locations)

    Input
    ---
    - room_dimension:       Array [length, width, height]   = room dimensions
    - abs_coef:             float                           = Sabine absorbtion coefficient
    - rate:                 optionnal, int                  = rate of the microphone (44100 default)
    - max_order:            optionnal, int                  = maximum reflection order in the image source model
    - sigma2_awgn:          optionnal, float                = variance of the additive white Gaussian noise added during simulation
    - display_room:         optionnal, bool                 = room display (False default)

    Output
    ---

    - room:                 ShoeBox                         = room

    """
    # Create an shoebox room
    room = pra.ShoeBox(
        room_dimension,
        fs=rate,
        max_order=max_order,
        absorption=abs_coef,
        sigma2_awgn=sigma2_awgn,
    )

    if display_room:
        fig, ax = room.plot()
        lim = 9
        ax.set_xlim([0, lim])
        ax.set_ylim([0, lim])
        ax.set_zlim([0, lim])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    return room


def compute_W(path, instrument, L=2048, pitch_min=21, pitch_max=109):
    """
    This function computes the PSD of the instrument notes from an audio file
    and returns the matrix W, that correspond to the dictionary of the instrument.


    Inputs:
    ------------------------------------------------------

    path : string
        path to the audio files of the instrument notes (str) - ex: './piano/' - 1 path per instrument

    L : int
        frame size (2048 default)

    pitch_min : int
        minimum pitch of the instrument (21 default)

    pitch_max : int
        maximum pitch of the instrument (109 default)


    Returns:
    ------------------------------------------------------

    W : numpy.ndarray (n_frequencies, n_notes)
        matrix of the instrument notes PSD

    notes : list (n_notes, n_samples)
        list of the instrument notes (audio signal)

    notes_psd : numpy.ndarray (n_notes, n_frequencies)
        matric of the instrument notes PSD

    freqs : list (n_frequencies)
        list of the frequencies of the PSD

    """
    notes = []
    notes_psd = []

    pitches = np.arange(pitch_min, pitch_max)

    for i, pitch in enumerate(pitches):
        xt, samplerate = librosa.load(path + str(int(pitch)) + instrument + ".wav", sr=None)
        freqs, Pxx_den = sp.signal.welch(xt, nfft=L)
        notes.append(xt)
        notes_psd.append(Pxx_den)

    notes_psd = np.asarray(notes_psd)

    W_piano = notes_psd.T

    return W_piano, notes, notes_psd, freqs, samplerate


def room_spectrogram_from_musdb(
    room,
    source_locations,
    source_dir,
    song_path,
    audio_length,
    L=2048,
    hop=512,
    rate=44100,
    display_audio=False,
    display_room=False
):

    """

    this function process a song from MUSDB18 dataset
    as a recording in a shoebox room (defined with its
    geometry, room absorption, signal locations and
    microphones locations) into a multichannel STFT.

    Input
    ---
    - room:                 room                                = room dimensions
    - source_locations:     Array [[3D locations],...]          = source locations
    - source_dir:           optionnal, directivity              = source directivity (None default)
    - song_path:            int                                 = number of the song
    - audio_length:         int                                 = length in second
    - L:                    optionnal, int                      = frame size (2048 default)
    - hop:                  optionnal, int                      = hop length (512 default)
    - rate:                 optionnal, int                      = signal rate (44100 default)
    - display_audio:        optionnal, bool                     = audio display (False default)
    - display_room:         optionnal, bool                     = room display (False default)

    Output
    ---

    - X:                    Array [4 x frequency x time_step]   = multichannel spectogram of the song
    - separate recordings:  Array [4 x 4 x time_step]           = recordings of each channel on each microphone separately
    - mics signals:         Array [4 x time_step]               = recordings of each microphones

    """
    path = song_path
    data, rate = stempeg.read_stems(path)
    channel_nb = room.n_mics

    X = []

    # Add sources
    for channel_source, source_loc in zip(range(1, len(data)), source_locations):
        signal_channel = librosa.core.to_mono(
            data[channel_source, : rate * audio_length, :].T
        )
        room.add_source(
            position=source_loc, directivity=source_dir, signal=signal_channel
        )

    if display_room:
        fig, ax = room.plot()
        lim = 9
        ax.set_xlim([0, lim])
        ax.set_ylim([0, lim])
        ax.set_zlim([0, lim])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Recordings
    separate_recordings = room.simulate(return_premix=True)
    mics_signals = np.sum(separate_recordings, axis=0)

    # STFT parameters
    win_a = pra.hamming(L)
    win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

    # Observation vector in the STFT domain
    for channel in range(channel_nb):

        Xn = pra.transform.stft.analysis(mics_signals[channel].T, L, hop, win=win_a)
        X.append(Xn)

    X = np.array(X)

    if display_audio:
        for microphone_n in range(channel_nb):
            display(ipd.Audio(mics_signals[microphone_n], rate=room.fs))

    return X, separate_recordings, mics_signals


def spectrogram_from_musdb(
    room_dimension,
    abs_coef,
    source_locations,
    microphone_locations,
    song_path,
    N_fft=512,
    Hop_length=512,
    Win_length=None,
    Display=False,
):

    """

    this function process a song from MUSDB18 dataset
    as a recording in a shoebox room (defined with its
    geometry, room absorption, signal locations and
    microphones locations) into a multichannel STFT.

    Inputs:

    source_locations
    microphone_locations (warning, locs in `np.c_` class)
    song: int (number of the song)
    n_fft: frequency steps (512 default)
    hop_length: hop length (51 default)
    win_length: window length

    Output:

    X: (M x frequency x time)

    """
    path = song_path
    data, rate = stempeg.read_stems(path)
    channel_nb, time_step, _ = data.shape
    X = []
    Xn = []

    # Create an shoebox room
    room = pra.ShoeBox(
        room_dimension, fs=44100, max_order=15, absorption=abs_coef, sigma2_awgn=1e-8
    )

    # Add sources
    for channel_source, source_loc in zip(range(1, len(data)), source_locations):
        signal_channel = librosa.core.to_mono(data[channel_source].T)
        room.add_source(source_loc, signal=signal_channel)

    # Add microphone array
    mic_array = pra.MicrophoneArray(microphone_locations, rate)
    room.add_microphone_array(mic_array)
    room.simulate()

    for microphone_n in range(channel_nb - 1):

        # Compute the STFT for each microphones
        microphone_output = room.mic_array.signals[microphone_n, :]
        Xstft = librosa.stft(
            microphone_output, n_fft=N_fft, hop_length=Hop_length, win_length=Win_length
        )
        Xn = np.abs(Xstft)

        if Display:
            display(ipd.Audio(microphone_output, rate=room.fs))

        # Concatenate Xn
        X.append(Xn)

    return X


def main():
    return


if __name__ == "__main__":
    main()
