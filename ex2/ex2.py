import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def add_frequency_band(audio_file, intervals):
    """
    Add a high-frequency band to the specified time intervals of an audio track.

    Parameters:
        audio_file (str): Path to the audio file.
        intervals (list of tuples): List of (start, end) time intervals (in seconds) 
                                    where the high-frequency band will be added.
    """
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Determine the frequency range for the marker (e.g., 10kHz below Nyquist to Nyquist)
    nyquist = sr / 2
    max_freq = nyquist
    min_freq = max(0, nyquist - 10000)  # 10kHz below Nyquist
    
    # Generate a high-frequency band
    def generate_band(duration, sr, min_freq, max_freq):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        band_signal = np.sin(2 * np.pi * min_freq * t) + np.sin(2 * np.pi * max_freq * t)
        band_signal /= np.max(np.abs(band_signal))  # Normalize to [-1, 1]
        return band_signal

    # Modify the audio
    y_modified = np.copy(y)
    for start, end in intervals:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        duration = end - start
        
        # Generate the band signal for the duration of the interval
        band_signal = generate_band(duration, sr, min_freq, max_freq)
        
        # Add the band to the corresponding audio segment
        length = min(len(band_signal), end_sample - start_sample)
        y_modified[start_sample:start_sample + length] += band_signal[:length]
    
    # Prevent clipping
    y_modified = np.clip(y_modified, -1.0, 1.0)
    
    # Save the modified audio
    output_file = f"{audio_file.rsplit('.', 1)[0]}_with_markers.wav"
    sf.write(output_file, y_modified, sr)
    print(f"Modified audio saved to {output_file}")

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



if __name__ == "__main__":
    # TASK 1

    input_audio = "Task 1/task1.wav"  # Replace with your input audio file
    output_audio = "Task 1/watermarked_task1.wav"  # Replace with your output file path

    add_frequency_band(input_audio, [(1,3), (10,13)])
    plot_spectrogram(f"{input_audio}_with_markers.wav")

    # Example usage
    # TASK 2
    # for i in range(9):
    #     audio_file_path = f"Task 2/{i}_watermarked.wav"  # Replace with your audio file path
    #     plot_spectrogram(audio_file_path)
