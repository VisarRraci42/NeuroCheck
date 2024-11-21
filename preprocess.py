import os
import numpy as np
import librosa

def load_audio_files(directory, sample_rate=22050):
    """
    Load all .wav files from the specified directory.

    Parameters:
        directory (str): Path to the directory containing .wav files.
        sample_rate (int): Target sampling rate.

    Returns:
        list of np.ndarray: List containing audio time series.
        list of str: Corresponding file names.
    """
    audio_data = []
    file_names = []
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            file_path = os.path.join(directory, file)
            # Load audio file
            y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
            # Optional: Trim silence or extract specific segments
            y, _ = librosa.effects.trim(y)
            audio_data.append(y)
            file_names.append(file)
    return audio_data, file_names

# Example usage
audio_directory = '/Users/visar/Desktop/rhd/minga-rhd-afflicted'
audio_data, file_names = load_audio_files(audio_directory)
