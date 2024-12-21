
from scipy.signal import stft, istft
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.io import wavfile
from sklearn.cluster import KMeans
from scipy.signal import find_peaks



def truncate_audio_frequencies(audio_file_path, start_freq, end_freq, output_file_path):
    """
    Truncates an audio file to retain only frequencies within the given range.
    
    :param audio_file_path: Path to the input audio file.
    :param start_freq: Start frequency of the range to retain (in Hz).
    :param end_freq: End frequency of the range to retain (in Hz).
    :param output_file_path: Path to save the truncated audio file.
    """
    # Read the audio file
    sample_rate, audio_data = wavfile.read(audio_file_path)

    # Handle stereo audio by converting to mono
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Perform STFT to transform the audio signal to the frequency domain
    frequencies, times, Zxx = stft(audio_data, fs=sample_rate)

    # Zero out frequencies outside the desired range
    Zxx[(frequencies < start_freq) | (frequencies > end_freq), :] = 0

    # Perform inverse STFT to transform back to the time domain
    _, truncated_audio = istft(Zxx, fs=sample_rate)

    # Normalize audio to prevent clipping
    truncated_audio = np.int16(truncated_audio / np.max(np.abs(truncated_audio)) * 32767)

    # Save the truncated audio to a new file
    wavfile.write(output_file_path, sample_rate, truncated_audio)
    print(f"Truncated audio saved to: {output_file_path}")