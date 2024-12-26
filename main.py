import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample
import librosa
import librosa.display
import librosa.feature
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

def plot_waveform(audio, fs, ax, filename=None):
    """
    Plots the audio waveform.

    Parameters:
        audio (numpy.ndarray): Audio signal.
        fs (int): Sampling frequency.
        ax (matplotlib.axes.Axes): Axes to plot on.
        filename (str, optional): Title label. Defaults to None.
    """
    time = np.linspace(0, len(audio) / fs, num=len(audio))
    ax.plot(time, audio, color='blue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform: {filename}" if filename else "Waveform")
    ax.grid(True)

def plot_spectrogram_with_pitch(audio, fs, ax, window_size_ms=20, hop_size_ms=10, filename=None):
    """
    Plots the spectrogram with a red pitch contour.

    Parameters:
        audio (numpy.ndarray): Audio signal.
        fs (int): Sampling frequency.
        ax (matplotlib.axes.Axes): Axes to plot on.
        window_size_ms (float, optional): Window size in milliseconds. Defaults to 20.
        hop_size_ms (float, optional): Hop size in milliseconds. Defaults to 10.
        filename (str, optional): Title label. Defaults to None.
    """
    # Convert window and hop sizes from ms to samples
    window_size = int(window_size_ms * fs / 1000)
    hop_length = int(hop_size_ms * fs / 1000)
    n_fft = window_size

    # Compute Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=window_size, window='hann')
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Extract pitch using pyworld
    audio_64 = audio.astype(np.float64)
    frame_period = (hop_length / fs) * 1000.0  # in ms
    f0, t_pitch = pw.harvest(audio_64, fs, f0_floor=50.0, f0_ceil=500.0, frame_period=frame_period)
    f0[f0 == 0] = np.nan  # Replace unvoiced frames with NaN

    # Time axis for spectrogram
    time_spec = np.arange(stft_db.shape[1]) * (hop_length / fs)

    # Plot spectrogram with default colormap
    img = librosa.display.specshow(stft_db, sr=fs, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram with Pitch: {filename}" if filename else "Spectrogram with Pitch")
    fig = plt.gcf()
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    # Overlay pitch contour in red
    ax.plot(t_pitch, f0, color='red', linewidth=2, label="Pitch Contour (F0)")
    ax.legend(loc="upper right")

def plot_mel_spectrogram(audio, fs, ax, window_size_ms=20, hop_size_ms=10, n_mels=64, fmax=None, filename=None):
    """
    Plots the Mel-spectrogram.

    Parameters:
        audio (numpy.ndarray): Audio signal.
        fs (int): Sampling frequency.
        ax (matplotlib.axes.Axes): Axes to plot on.
        window_size_ms (float, optional): Window size in milliseconds. Defaults to 20.
        hop_size_ms (float, optional): Hop size in milliseconds. Defaults to 10.
        n_mels (int, optional): Number of Mel bands. Defaults to 64.
        fmax (float, optional): Maximum frequency. Defaults to None (uses sr/2).
        filename (str, optional): Title label. Defaults to None.
    """
    # Convert window and hop sizes from ms to samples
    window_size = int(window_size_ms * fs / 1000)
    hop_length = int(hop_size_ms * fs / 1000)
    n_fft = max(256, window_size)  # Ensure n_fft is at least 256 for better frequency resolution

    # Compute Mel-spectrogram and convert to dB
    S_mel = librosa.feature.melspectrogram(
        y=audio, 
        sr=fs, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=window_size,
        window='hann', 
        n_mels=n_mels,
        fmax=fmax if fmax else fs / 2
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    # Time axis for Mel-spectrogram
    time_mel = np.arange(S_mel_db.shape[1]) * (hop_length / fs)

    # Plot Mel-spectrogram with default colormap
    img = librosa.display.specshow(S_mel_db, sr=fs, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax, fmax=fmax if fmax else fs / 2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Frequency (Hz)")
    ax.set_title(f"Mel-Spectrogram: {filename}" if filename else "Mel-Spectrogram")
    fig = plt.gcf()
    fig.colorbar(img, ax=ax, format="%+2.f dB")

def plot_energy_rms(audio, fs, ax, window_size_ms=20, hop_size_ms=10, filename=None):
    """
    Plots Energy and RMS of the audio signal.

    Parameters:
        audio (numpy.ndarray): Audio signal.
        fs (int): Sampling frequency.
        ax (matplotlib.axes.Axes): Axes to plot on.
        window_size_ms (float, optional): Window size in milliseconds. Defaults to 20.
        hop_size_ms (float, optional): Hop size in milliseconds. Defaults to 10.
        filename (str, optional): Title label. Defaults to None.
    """
    # Convert window and hop sizes from ms to samples
    window_size = int(window_size_ms * fs / 1000)
    hop_length = int(hop_size_ms * fs / 1000)

    # Compute RMS and Energy
    rms = librosa.feature.rms(y=audio, frame_length=window_size, hop_length=hop_length)[0]
    energy = rms ** 2

    # Time axis for Energy and RMS
    time_energy = np.arange(len(rms)) * (hop_length / fs)

    # Plot Energy and RMS
    ax.plot(time_energy, energy, color='green', label='Energy')
    ax.plot(time_energy, rms, color='orange', label='RMS')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Energy and RMS: {filename}" if filename else "Energy and RMS")
    ax.legend(loc="upper right")
    ax.grid(True)

def plot_audio_analysis(audio, fs, filename=None):
    """
    Plots a figure with 4 subplots:
    1. Waveform
    2. Spectrogram with Pitch Contour
    3. Mel-Spectrogram
    4. Energy and RMS

    Parameters:
        audio (numpy.ndarray): Audio signal.
        fs (int): Sampling frequency.
        filename (str, optional): Title label. Defaults to None.
    """
    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    # Plot each component
    plot_waveform(audio, fs, axs[0, 0], filename)
    plot_spectrogram_with_pitch(audio, fs, axs[0, 1], filename=filename)
    plot_mel_spectrogram(audio, fs, axs[1, 0], n_mels=64, fmax=fs/2, filename=filename)
    plot_energy_rms(audio, fs, axs[1, 1], filename=filename)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def time_stretch_phase_vocoder(audio, fs, speed_factor=1.5, window_size_ms=20, hop_size_ms=10):
    """
    Increases the speed of the audio by a given factor while preserving the pitch using a phase vocoder.

    Parameters:
        audio (numpy.ndarray): Original audio signal.
        fs (int): Sampling frequency.
        speed_factor (float, optional): Factor by which to speed up the audio. Default is 1.5.
        window_size_ms (float, optional): Window size in milliseconds. Defaults to 20.
        hop_size_ms (float, optional): Hop size in milliseconds. Defaults to 10.

    Returns:
        audio_stretched (numpy.ndarray): Time-stretched audio signal.
    """
    # Calculate the time stretch factor
    time_stretch_factor = speed_factor
    print(f"\nApplying time stretching: Speeding up by {speed_factor}x (time stretch factor = {time_stretch_factor})")

    # Convert window and hop sizes from ms to samples
    window_size = int(window_size_ms * fs / 1000)
    hop_length = int(hop_size_ms * fs / 1000)
    n_fft = window_size

    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=window_size, window='hann')
    magnitude, phase = np.abs(stft), np.angle(stft)
    print("Original STFT computed.")

    # Apply phase vocoder
    stft_stretched = librosa.phase_vocoder(stft, rate=time_stretch_factor, hop_length=hop_length)
    print("Phase vocoder applied.")

    # Calculate magnitude and phase of the stretched STFT
    magnitude_stretched, phase_stretched = np.abs(stft_stretched), np.angle(stft_stretched)
    print("Magnitude and phase of the stretched STFT calculated.")

    # Invert STFT to get time-stretched audio
    audio_stretched = librosa.istft(stft_stretched, hop_length=hop_length, win_length=window_size, window='hann')
    print(f"Time-stretched audio duration: {len(audio_stretched)/fs:.2f} seconds")

    return audio_stretched, magnitude_stretched, phase_stretched

def plot_time_domain_comparison(original_audio, stretched_audio, fs, speed_factor=1.5):
    """
    Plots the original and stretched audio waveforms side by side.
    """
    plt.figure(figsize=(18, 6))

    # Original Audio Waveform
    plt.subplot(1, 2, 1)
    time_original = np.linspace(0, len(original_audio) / fs, num=len(original_audio))
    plt.plot(time_original, original_audio, color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original Audio Waveform")
    plt.grid(True)

    # Stretched Audio Waveform
    plt.subplot(1, 2, 2)
    time_stretched = np.linspace(0, len(stretched_audio) / fs, num=len(stretched_audio))
    plt.plot(time_stretched, stretched_audio, color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Stretched Audio Waveform ({speed_factor}x Speed)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_spectral_domain_comparison(original_audio, stretched_audio, fs, speed_factor=1.5):
    """
    Plots the normalized frequency spectra (FFT) of the original and stretched audio side by side.
    """
    # Compute FFT for Original Audio
    N_orig = len(original_audio)
    fft_orig = np.fft.fft(original_audio)
    fft_orig = fft_orig[:N_orig//2]  # Take positive frequencies
    freq_orig = np.linspace(0, fs/2, num=len(fft_orig))
    magnitude_orig = np.abs(fft_orig) / N_orig  # Normalize by number of samples

    # Compute FFT for Stretched Audio
    N_stretched = len(stretched_audio)
    fft_stretched = np.fft.fft(stretched_audio)
    fft_stretched = fft_stretched[:N_stretched//2]  # Take positive frequencies
    freq_stretched = np.linspace(0, fs/2, num=len(fft_stretched))
    magnitude_stretched = np.abs(fft_stretched) / N_stretched  # Normalize by number of samples

    plt.figure(figsize=(18, 6))

    # Original Audio Spectrum
    plt.subplot(1, 2, 1)
    plt.plot(freq_orig, magnitude_orig, color='blue')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.title("Original Audio Spectrum")
    plt.grid(True)
    plt.xlim(0, fs/2)

    # Stretched Audio Spectrum
    plt.subplot(1, 2, 2)
    plt.plot(freq_stretched, magnitude_stretched, color='orange')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.title(f"Stretched Audio Spectrum ({speed_factor}x Speed)")
    plt.grid(True)
    plt.xlim(0, fs/2)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # question 1(a)
    counting_fs, counting_audio = load_audio("speech_recording.wav")

    # question 1(b) - Resample to 32 kHz
    resample_audio_32k, fs_32k = resample_audio(counting_audio, counting_fs, 32000)
    save_audio("resample_audio_32k.wav", fs_32k, resample_audio_32k)

    # question 1(c) - Two methods to get 16 kHz
    resample_audio_manual, fs_16k = downsample_manual(resample_audio_32k)
    resample_audio_16k, fs_16k = resample_audio(resample_audio_32k, fs_32k, 16000)
    save_audio("resample_audio_16k.wav", fs_16k, resample_audio_16k)
    save_audio("resample_audio_manual.wav", fs_16k, resample_audio_manual)


    # question 1(d) - plots
    # 1. plot_waveform
    # 2. plot_spectrogram_with_pitch
    # Explanation for missing timeframes in pitch contour:
    # Missing timeframes occur during unvoiced segments (like silence or noise) where no clear pitch is detected by the algorithm.
    # 3. plot_mel_spectrogram
    # 4. plot_energy_rms
    # 5. plot_audio_analysis
    plot_audio_analysis(resample_audio_16k, fs_16k, filename="resample_audio_16k.wav")
    plot_audio_analysis(resample_audio_manual, fs_16k, filename="resample_audio_manual.wav")
    # add answers to the questions in the report

    # question 2(a), 2(b) - Add noise
    noise_audio = add_noise(resample_audio_16k)
    save_audio("noisy_speech.wav", fs_16k, noise_audio)

    # question 2(c) - plots
    plot_audio_analysis(resample_audio_16k, fs_16k, filename="resample_audio_16k.wav")
    fs_noise , stationary_noise  = load_audio("stationary_noise.wav")
    plot_audio_analysis(stationary_noise, fs_noise, filename="stationary_noise.wav")
    plot_audio_analysis(noise_audio, fs_16k, filename="noisy_speech.wav")

    # question 5 time-stretch the audio
    speed_factor = 1.5
    audio_stretched, magnitude_stretched, phase_stretched = time_stretch_phase_vocoder(resample_audio_16k, fs_16k, speed_factor=speed_factor)
    stretched_filename = f'stretched_audio_{speed_factor}x.wav'
    save_audio(stretched_filename, fs_16k, audio_stretched)

    
    print("\n### Time Domain Comparison ###")
    plot_time_domain_comparison(resample_audio_16k, audio_stretched, fs_16k)
    print("\n### Spectral Domain Comparison ###")
    plot_spectral_domain_comparison(resample_audio_16k, audio_stretched, fs_16k)
  