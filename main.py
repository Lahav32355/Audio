import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy.signal import resample
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#Todo:Record yourself speaking
def record_audio(filename, duration=10, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, recording)
    print(f"Recording saved to {filename}")


def save_audio(filename, fs, data):
    wavfile.write(filename, fs, data.astype(np.float32))
    print(f"Resampled audio saved to {filename}")


def load_audio(filename):
    # Load the audio file we Record
    sampling_freq, data = wavfile.read(filename)
    print(f"Sampling Frequency: {sampling_freq} Hz")

    # Check if stereo
    if len(data.shape) == 2:
        print("Audio is Stereo")
        mono_data = data[:, 0]  #Select the first channel
    else:
        print("Audio is Mono")
        mono_data = data

    return sampling_freq, mono_data


def resample_audio(audio, original_fs, target_fs=32000):
    # Cast audio to np.float32
    audio_float32 = audio.astype(np.float32)
    # Calculate the number of samples after resampling
    duration = len(audio_float32) / original_fs
    num_samples_resampled = int(duration * target_fs)
    # Perform resampling
    audio_resampled = resample(audio_float32, num_samples_resampled)
    print("Resampling completed.")
    return audio_resampled , target_fs

def downsample_manual(audio , target_fs = 16000):
    downsample_audio = audio[::2]
    return downsample_audio , target_fs

#Todo:
def plotaudio():
    pass

#Todo:
def add_noise(audio):
    noise_fs, noise = load_audio("stationary_noise.wav")
    resampled_noise = resample_audio(noise, noise_fs, 16000)
    if len(resampled_noise) < len(audio):
        return  audio[:len(resampled_noise)] + resampled_noise
    return audio + resampled_noise[:len(audio)]  # Add noise to the audio






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





#Todo:Cheking the order
if __name__ == "__main__":
    # question 1(a)
    counting_fs, counting_audio = load_audio("speech_recording.wav")

    # question 1(b)
    resample_audio_32k ,fs_32k = resample_audio(counting_audio , counting_fs)

    # question 1(c)
    resample_audio_manual , fs_16k = downsample_manual(resample_audio_32k)
    resample_audio_16k , fs_16k = resample_audio(counting_audio , counting_fs , 16000)





    