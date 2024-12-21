import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
from pydub.generators import Sine
from scipy.io import wavfile

from scipy.signal import stft, istft
from scipy.fftpack import fft
import os

def add_continuous_tone(audio_file_path, frequency, output_file_path):
    """
    Adds a continuous sound at the given frequency to an audio file.
    
    :param audio_file_path: Path to the input audio file.
    :param frequency: Frequency of the continuous sound to add (in Hz).
    :param output_file_path: Path to save the modified audio file.
    """
    # Load the original audio
    original_audio = AudioSegment.from_file(audio_file_path)
    
    # Get the duration of the original audio in milliseconds
    duration_ms = len(original_audio)
    
    # Generate a sine wave tone for the duration of the original audio
    sine_wave = Sine(frequency).to_audio_segment(duration=duration_ms)
    
    # Adjust the volume of the sine wave (optional, default is 0dB)
    sine_wave = sine_wave - 10  # Reduce volume by 10dB
    
    # Overlay the sine wave onto the original audio
    combined_audio = original_audio.overlay(sine_wave)
    
    # Export the modified audio
    combined_audio.export(output_file_path, format="wav")
    print(f"Modified audio saved to: {output_file_path}")

def plot_spectrogram(audio_file):
    """
    Plots the spectrogram of an audio file.
    
    Parameters:
        audio_file (str): Path to the audio file.
    """
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Compute the Short-Time Fourier Transform (STFT)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(f"{audio_file}_specto.png")



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

def classify_audio(num):
    """
    since hard coding is allowed and we are not require to make general classification:
    Group 1: 1,2,3
    Group 2: 4,5,6
    Group 3: 6,7,8
    """
    
    if num < 3:
        return 1
    if num < 6:
        return 2
    return 3



def plot_signal(audio_path, output_file=None):
    """
    Save the time-domain signal of an audio file as a plot image.
    :param audio_path: Path to the audio file
    :param output_dir: Directory to save the plot, defaults to audio file's directory
    """
    # Read the audio file
    signal, sample_rate = sf.read(audio_path)
    duration = len(signal) / sample_rate
    time = np.linspace(0, duration, len(signal))
    
    # If stereo, convert to mono
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
        
    # Save the time-domain signal plot
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, color='b')
    plt.title("Time-Domain Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Time-domain plot saved to: {output_file}")


def plot_freq(audio_path, output_file):
    """
    Save the frequency-domain representation of an audio file as a plot image.
    :param audio_path: Path to the audio file
    :param output_dir: Directory to save the plot, defaults to audio file's directory
    """
    # Read the audio file
    signal, sample_rate = sf.read(audio_path)
    print(f"sample rate of {audio_path} is {sample_rate}")
    
    # If stereo, convert to mono
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    
    # Apply FFT
    N = len(signal)
    freq = np.linspace(0, sample_rate / 2, N // 2)
    fft_values = fft(signal)
    magnitude = np.abs(fft_values)[:N // 2]  # Keep only the positive half
        
    # Save the frequency-domain plot
    plt.figure(figsize=(10, 4))
    plt.plot(freq, magnitude, color='r')
    plt.title(f"Frequency-Domain plot for {audio_path}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Frequency-domain plot saved to: {output_file}")


def task_one():
    input_audio = "Task 1/task1.wav"  # Replace with your input audio file
    def output_audio(x):
        return f"Task 1/watermarked_task1_{x}.wav"  # Replace with your output file path

    add_continuous_tone(input_audio, 1000, output_audio("bad"))
    add_continuous_tone(input_audio, 19500, output_audio("good"))

    plot_spectrogram(output_audio("bad"))
    plot_spectrogram(output_audio("good"))

def task_two():
    for i in range(9):
        input_audio = f"Task 2/{i}_watermarked.wav"
        truncate_audio_frequencies(input_audio,16000,20000, f"{input_audio}_truncated.wav")
        plot_spectrogram(f"{input_audio}_truncated.wav")
        print(f"file {input_audio} belongs to group: {classify_audio(i)}")

def task_three():
    for i in range(2):
        input_audio = f"Task 3/task3_watermarked_method{i+1}.wav"
        plot_signal(input_audio, f"{input_audio}_signal.png")
        plot_freq(input_audio,f"{input_audio}_freq.png")
        plot_spectrogram(input_audio)


if __name__ == "__main__":
    # TASK 1
    # task_one()

    # # # # TASK 2
    # task_two()
    
    # # TASK 3 
    task_three()
