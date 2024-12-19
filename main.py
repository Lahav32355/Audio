import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy.signal import resample
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np


def plot_audio(audio, sr, ax):
    """Plots the raw audio waveform."""
    time = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time, audio)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Amplitude")


def plot_spectrogram(audio, sr, ax):
    """Plots the spectrogram with the pitch contour overlaid."""
    # Calculate the spectrogram
    hop_length = int(sr * 0.01)  # 10ms
    n_fft = int(sr * 0.02)  # 20ms
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Plot the spectrogram
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title("Spectrogram")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")

    # Pitch contour using librosa.pyin
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length
    )

    # Overlay pitch contour
    valid_frames = ~np.isnan(f0)  # Non-NaN values indicate voiced frames
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    ax.plot(times[valid_frames], f0[valid_frames], color='red', label='Pitch Contour')
    ax.legend()


def plot_mel_spectrogram(audio, sr, ax):
    """Plots the Mel-spectrogram."""
    hop_length = int(sr * 0.01)  # 10ms
    n_fft = int(sr * 0.02)  # 20ms
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Mel-Spectrogram")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")


def plot_energy_and_rms(audio, sr, ax):
    """Plots energy and RMS."""
    frame_length = int(sr * 0.02)  # 20ms
    hop_length = int(sr * 0.01)  # 10ms

    # Energy
    energy = np.array([
        np.sum(np.abs(audio[i:i + frame_length]) ** 2)
        for i in range(0, len(audio) - frame_length + 1, hop_length)
    ])

    # RMS
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()

    # Time axis for plotting
    times = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=hop_length)

    ax.plot(times, energy, label="Energy", alpha=0.7)
    ax.plot(times, rms, label="RMS", alpha=0.7)
    ax.set_title("Energy and RMS")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Amplitude")
    ax.legend()


def plot_audio_analysis(audio, sr):
    """Main function to plot all subplots together."""
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    plot_audio(audio, sr, axs[0])
    plot_spectrogram(audio, sr, axs[1])
    plot_mel_spectrogram(audio, sr, axs[2])
    plot_energy_and_rms(audio, sr, axs[3])

    plt.tight_layout()
    plt.show()



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
    resampled_noise, _ = resample_audio(noise, noise_fs, 16000)
    if len(resampled_noise) < len(audio):
        return audio[:len(resampled_noise)] + resampled_noise
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

    # question 2
    new_audio = add_noise(resample_audio_16k)
    # save_audio("noisy_speech.wav", counting_fs, new_audio)
    plot_audio_analysis(counting_audio, fs_16k)





