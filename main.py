import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy.signal import resample
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def record_audio(filename, duration=10, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)  # Stereo
    sd.wait()  # Wait until recording is finished
    write(filename, fs, recording)  # Save as WAV file
    print(f"Recording saved to {filename}")


def load_audio(filename):
    # Load the audio file
    sampling_freq, data = wavfile.read(filename)
    print(f"Original Sampling Frequency: {sampling_freq} Hz")

    # Check if stereo
    if len(data.shape) == 2:
        print("Audio is Stereo")
        # Option 1: Select the first channel
        mono_data = data[:, 0]
    else:
        print("Audio is Mono")
        mono_data = data

    print(f"Sampling Frequency: {sampling_freq} Hz")

    return sampling_freq, mono_data


def resample_audio(audio, original_fs, target_fs=32000):
    # Cast audio to np.float32
    audio_float32 = audio.astype(np.float32)
    print("Audio cast to np.float32.")

    # Calculate the number of samples after resampling
    duration = len(audio_float32) / original_fs
    num_samples_resampled = int(duration * target_fs)
    print(f"Resampling from {original_fs} Hz to {target_fs} Hz.")
    print(f"Original number of samples: {len(audio_float32)}")
    print(f"Resampled number of samples: {num_samples_resampled}")

    # Perform resampling
    audio_resampled = resample(audio_float32, num_samples_resampled)
    print("Resampling completed.")

    return audio_resampled

def save_audio(filename, fs, data):
    # Ensure the data is in the correct format (float32)
    wavfile.write(filename, fs, data.astype(np.float32))
    print(f"Resampled audio saved to {filename}")



def plot_audio_and_spectrogram(audio_path):
    """
    Given an audio file, plot the audio waveform and its spectrogram,
    including Nyquist frequency validation.

    Parameters:
    - audio_path: Path to the input audio file.

    Returns:
    - None
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)  # Keep original sampling rate
    duration = len(y) / sr  # Duration of the audio in seconds

    # Compute Short-Time Fourier Transform (STFT) for the spectrogram
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert to decibels

    # Nyquist Frequency
    f_max = sr / 2  # Nyquist frequency is half the sampling rate

    # Create time and frequency arrays for plotting
    time = np.linspace(0, duration, num=len(y))
    freq_bins = librosa.fft_frequencies(sr=sr)

    # Plot the figure with 4 subplots
    fig, ax = plt.subplots(4, 1, figsize=(12, 12))

    # Plot (i) Audio waveform
    ax[0].plot(time, y, label="Audio waveform")
    ax[0].set_title("Audio Waveform")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend(loc="upper right")

    # Plot (ii) Spectrogram
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear', ax=ax[1])
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    ax[1].set_title("Spectrogram")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")

    # Validate and mark Nyquist frequency
    ax[1].axhline(f_max, color='r', linestyle='--', label=f"Nyquist Frequency (Fmax = {f_max:.1f} Hz)")
    ax[1].legend(loc="upper right")

    # Plot (iii) Magnitude spectrum (Frequency axis validation)
    spectrum = np.abs(D).mean(axis=1)  # Average magnitude across time
    ax[2].plot(freq_bins, spectrum, label="Average Magnitude Spectrum")
    ax[2].set_title("Magnitude Spectrum")
    ax[2].set_xlabel("Frequency (Hz)")
    ax[2].set_ylabel("Magnitude")
    ax[2].axvline(f_max, color='r', linestyle='--', label=f"Nyquist Frequency (Fmax = {f_max:.1f} Hz)")
    ax[2].legend(loc="upper right")

    # Plot (iv) Zoomed-in Spectrogram (Focus on frequencies < Fmax)
    zoom_freq_bins = freq_bins[freq_bins < f_max]
    zoom_S_db = S_db[:len(zoom_freq_bins), :]  # Focus only on the frequency bins below Fmax
    img_zoom = librosa.display.specshow(zoom_S_db, sr=sr, x_axis='time', y_axis='linear', ax=ax[3])
    fig.colorbar(img_zoom, ax=ax[3], format="%+2.0f dB")
    ax[3].set_title("Zoomed-In Spectrogram (Frequencies < Fmax)")
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Frequency (Hz)")

    # Display the figure
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    record_audio("speech_recording.wav")
    #fs_original, audio_original = load_audio("speech_recording.wav")
    #plot_audio_and_spectrogram("speech_recording.wav")

    # Load the audio file
    #sampling_freq, data = wavfile.read('speech_recording.wav')

    #fs_target = 32000  # 32 kHz
    #audio_resampled = resample_audio(audio_original, fs_original, target_fs=fs_target)

    # (Optional) Save the resampled audio to a new file
    #save_audio("speech_recording_resampled.wav", fs_target, audio_resampled)