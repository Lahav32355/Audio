import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np

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

def plots_audio(audio, fs):
    time = np.linspace(0, len(audio) / fs, num=len(audio))
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, label="Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.grid()
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    # question 1(a)
    counting_fs, counting_audio = load_audio("speech_recording.wav")

    # question 1(b) - Resample to 32 kHz
    resample_audio_32k, fs_32k = resample_audio(counting_audio, counting_fs, 32000)
    #save_audio("resample_audio_32k.wav", fs_32k, resample_audio_32k)

    # question 1(c) - Two methods to get 16 kHz
    resample_audio_manual, fs_16k = downsample_manual(resample_audio_32k)
    resample_audio_16k, fs_16k = resample_audio(resample_audio_32k, fs_32k, 16000)

    # question 2(a), 2(b) - Add noise
    noise_audio = add_noise(resample_audio_16k)
    # Normalize and save noisy audio
    #save_audio("noisy_speech.wav", fs_16k, noise_audio)

    # Optional Plots:
    plots_audio(resample_audio_16k, fs_16k)
    noise_fs, noise = load_audio("stationary_noise.wav")
    plots_audio(noise, noise_fs)
    plots_audio(noise_audio, fs_16k)
