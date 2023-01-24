def main():
    return

if __name__ == "__main__":
    main()

def files_MUSDB(path_in):

    """ 
    this function take the path to the MUSDB dataset as input
    and creates a list of the address of the songs, and a list of the song's name

    Inputs: 
    
    MUSDB18_path (address of the song folder)

    Output:

    files_in: list of song's path
    files_title: list of song's names

    """
    files_in = []
    files_title = []

    # extract files 
    for r, _ , f in os.walk(path_in):
        for file in f:
            if '.mp4' in file:

                # file address
                files_in.append(os.path.join(r, file))
                # file author + song
                files_title.append(file[:-9])
                
    files_in.sort()

    return(files_in, files_title)
    
def spectrogram_from_musdb(room_dimension, 
    abs_coef, 
    source_locations, 
    microphone_locations, 
    song_path, 
    N_fft=512, 
    Hop_length=512, 
    Win_length=None,
    Display = False):

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
    room = pra.ShoeBox(room_dimension, fs=44100, max_order=15, absorption=abs_coef, sigma2_awgn=1e-8)

    # Add sources
    for channel_source, source_loc in zip(range(1,len(data)), source_locations):
        signal_channel = librosa.core.to_mono(data[channel_source].T)
        room.add_source(source_loc, signal=signal_channel)

    # Add microphone array
    mic_array = pra.MicrophoneArray(microphone_locations, rate)
    room.add_microphone_array(mic_array)
    room.simulate()

    for microphone_n in range(channel_nb - 1) :

        # Compute the STFT for each microphones
        microphone_output = room.mic_array.signals[microphone_n,:]
        Xstft = librosa.stft(microphone_output, n_fft=N_fft, hop_length=Hop_length, win_length=Win_length)
        Xn = np.abs(Xstft)

        if Display :
            display(ipd.Audio(microphone_output, rate=room.fs))

        # Concatenate Xn
        X.append(Xn)

    return (X)