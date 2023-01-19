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
    
def MUSDB_data(song_path, Stereo = False, N_fft=512, Hop_length=512, Win_length=None):
    """ 
    this function process a song from MUSDB18 dataset
    into a multichannel STFT

    Inputs:

    song: int (number of the song)
    MUSDB18_path: character chain (address of the song folder)
    Stereo: boolean
    n_fft: frequency steps (512 default)
    hop_length: hop length (51 default)
    win_length: window length

    Output:

    X: (M x frequency x time)

    """
    path = song_path
    data, _ = stempeg.read_stems(path)
    channel_nb, time_step, _ = data.shape
    X = []
    Xn = []

    if Stereo == True :
        for channel in range(1, channel_nb) :

            # Compute the STFT for each channel
            data_n_left = S[channel].T[0]
            data_n_right = S[channel].T[1]

            Xstft_left = librosa.stft(data_n_left, n_fft=N_fft, hop_length=Hop_length, win_length=Win_length)
            Xstft_left = librosa.stft(data_n_left, n_fft=N_fft, hop_length=Hop_length, win_length=Win_length)

            Xn_left = np.abs(Xstft_left)
            Xn_right = np.abs(Xstft_left)

            # Concatenate Xn
            X.append(Xn_left)
            X.append(Xn_right)


    elif Stereo == False :
        for channel in range(1, channel_nb) :

            # Compute the STFT for each channel
            data_n = librosa.core.to_mono(S[channel].T)
            Xstft = librosa.stft(data_n, n_fft=N_fft, hop_length=Hop_length, win_length=Win_length)
            Xn = np.abs(Xstft)

            # Concatenate Xn
            X.append(Xn)

    return X