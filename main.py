import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample
import librosa
import librosa.display
import librosa.feature
import matplotlib
matplotlib.use("TkAgg")  # You can also try "QtAgg"
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw

def normalize_audio(data):
    """Normalize audio data to the range [-1,1]."""
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return data


def float32_to_int16(data):
    """Convert float32 audio in [-1,1] to int16."""
    return (data * 32767.0).astype(np.int16)


def save_audio(filename, fs, data):
    """
    Save normalized float32 data as int16 WAV.
    Assumes data is float32 in [-1,1].
    """
    data = normalize_audio(data)
    data_int16 = float32_to_int16(data)
    wavfile.write(filename, fs, data_int16)
    print(f"Audio saved to {filename} as int16")


def load_audio(filename):
    sampling_freq, data = wavfile.read(filename)
    print(f"Sampling Frequency: {sampling_freq} Hz")

    # Convert to mono if stereo
    if len(data.shape) == 2:
        print("Audio is Stereo, taking the left channel for mono")
        data = data[:, 0]
    else:
        print("Audio is Mono")

    # Convert int16 to float32 and assume range is [-32768, 32767]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    # If float32, we assume it's already in [-1, 1].

    return sampling_freq, data


def resample_audio(audio, original_fs, target_fs=32000):
    duration = len(audio) / original_fs
    num_samples_resampled = int(duration * target_fs)
    if num_samples_resampled <= 0:
        raise ValueError("No samples to resample. Check input audio or sampling rates.")
    audio_resampled = resample(audio, num_samples_resampled)
    print("Resampling completed.")
    return audio_resampled, target_fs


def downsample_manual(audio, target_fs=16000):
    # Simple downsampling by factor of 2
    downsampled_audio = audio[::2]
    return downsampled_audio, target_fs


def add_noise(audio):
    noise_fs, noise = load_audio("stationary_noise.wav")
    resampled_noise, _ = resample_audio(noise, noise_fs, 16000)

    min_len = min(len(audio), len(resampled_noise))
    noisy = audio[:min_len] + resampled_noise[:min_len]
    return noisy


def vad(audio, frame_length=1024, hop_length=512, energy_threshold=0.02):
    # Compute the short-time energy
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Identify speech and noise frames
    speech_frames = [i for i, e in enumerate(energy) if e > energy_threshold]
    noise_frames = [i for i, e in enumerate(energy) if e <= energy_threshold]

    return speech_frames, noise_frames


def spectral_subtraction(audio, frame_length=1024, hop_length=512, energy_threshold=0.02):
    stft_audio = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    magnitude, phase = np.abs(stft_audio), np.angle(stft_audio)

    _, noise_frames = vad(audio, frame_length, hop_length, energy_threshold)

    ## Todo - Plot the energy of the audio signal
    # plot_energy(audio, 16000, "Sound With Noise", False, energy_threshold)

    noise_spectrum = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
    enhanced_magnitude = np.maximum(magnitude - noise_spectrum, 0)
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)

    return enhanced_audio


def auto_gain_control(audio, sampling_rate, desired_rms_dB, noise_floor_dB=-34, window_duration=1.0):
    # Compute desired RMS in linear scale
    desired_rms = 10 ** (desired_rms_dB / 20)

    # Compute frame length and hop length in samples
    frame_length = int(window_duration * sampling_rate)
    hop_length = frame_length // 2  # 50% overlap

    # Compute RMS energy for each frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Convert RMS to dB
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))

    # Determine scaling factors
    scaling_factors = np.zeros_like(rms_db)
    for i, frame_rms_db in enumerate(rms_db):
        if frame_rms_db < noise_floor_dB:
            scaling_factors[i] = 0  # Mute noise
        else:
            scaling_factors[i] = desired_rms / (10 ** (frame_rms_db / 20))

    # Apply scaling factors sequentially
    agc_audio = np.zeros_like(audio)
    for i, gain in enumerate(scaling_factors):
        start = i * hop_length
        end = start + frame_length
        if end > len(audio):
            end = len(audio)
        agc_audio[start:end] += audio[start:end] * gain

    # Normalize to avoid clipping
    max_amplitude = np.max(np.abs(agc_audio))
    if max_amplitude > 1.0:
        agc_audio = agc_audio / max_amplitude

    return agc_audio, scaling_factors


def plot_results(original_audio, amplified_audio, scaling_factors, sample_rate):
    time_axis = np.arange(len(original_audio)) / sample_rate

    plt.figure(figsize=(12, 8))

    # Plot original audio
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, original_audio, label="Original Audio", color="blue", alpha=0.7)
    plt.title("Original Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Plot amplified audio
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, amplified_audio, label="Amplified Audio", color="orange", alpha=0.7)
    plt.title("Amplified Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Plot scaling factors
    time_scaling = np.arange(len(scaling_factors)) * (len(original_audio) / len(scaling_factors)) / sample_rate
    plt.subplot(3, 1, 3)
    plt.plot(time_scaling, scaling_factors, label="Scaling Factors", color="green", alpha=0.7)
    plt.title("Scaling Factors vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Scaling Factor")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plots_audio(audio, fs, filename=None):

    time = np.linspace(0, len(audio) / fs, num=len(audio))
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, label="Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    if filename is not None:
        plt.title(f"Audio Waveform for: {filename}")
    else:
        plt.title("Audio Waveform")

    plt.grid()
    plt.legend(loc="upper right")
    plt.show()


def plot_spectrogram_with_pitch(audio, fs, filename=None):
    n_fft = 1024
    hop_length = 256
    D = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(D, ref=np.max)

    # Convert audio to float64 for pyworld
    audio_64 = audio.astype(np.float64)
    frame_period = (hop_length / fs) * 1000.0  # in ms
    f0, t = pw.harvest(audio_64, fs, f0_floor=50.0, f0_ceil=500.0, frame_period=frame_period)
    f0[f0 == 0] = np.nan

    time_stft = np.arange(S_db.shape[1]) * (hop_length / fs)

    plt.figure(figsize=(10, 6))
    plt.imshow(S_db, aspect='auto', origin='lower',
               extent=[time_stft[0], time_stft[-1], 0, fs/2], cmap='magma')
    plt.colorbar(format="%+2.f dB")

    plt.plot(t, f0, color='red', linewidth=2, label="Pitch contour (F0)")

    if filename is not None:
        plt.title(f"Spectrogram & Pitch for: {filename}")
    else:
        plt.title("Spectrogram & Pitch")

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend(loc="upper right")
    plt.show()


