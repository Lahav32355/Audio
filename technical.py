import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import os


def normalize_audio(data):
    # Normalize audio data to the range [-1,1].
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return data


def float32_to_int16(data):
    # Convert float32 audio in [-1,1] to int16.
    return (data * 32767.0).astype(np.int16)


def save_audio(filename, fs, data):
    # Save normalized float32 data as int16 WAV.
    # Assumes data is float32 in [-1,1].
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
    time = np.linspace(0, len(audio) / fs, num=len(audio))
    ax.plot(time, audio, color='blue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform: {filename}" if filename else "Waveform")
    ax.grid(True)


def plot_spectrogram_with_pitch(audio, fs, ax, window_size_ms=20, hop_size_ms=10, filename=None):
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
    img = librosa.display.specshow(S_mel_db, sr=fs, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax,
                                   fmax=fmax if fmax else fs / 2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Frequency (Hz)")
    ax.set_title(f"Mel-Spectrogram: {filename}" if filename else "Mel-Spectrogram")
    fig = plt.gcf()
    fig.colorbar(img, ax=ax, format="%+2.f dB")


def plot_energy_rms(audio, fs, ax, window_size_ms=20, hop_size_ms=10, filename=None):
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
    plot_mel_spectrogram(audio, fs, axs[1, 0], n_mels=64, fmax=fs / 2, filename=filename)
    plot_energy_rms(audio, fs, axs[1, 1], filename=filename)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


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

    noise_spectrum = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
    enhanced_magnitude = np.maximum(magnitude - noise_spectrum, 0)
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)

    return enhanced_audio


def auto_gain_control(audio, sampling_rate, desired_rms_dB, noise_floor_dB=-34, window_duration=1.0):
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
    print(f"Time-stretched audio duration: {len(audio_stretched) / fs:.2f} seconds")

    return audio_stretched, magnitude_stretched, phase_stretched


def plot_audio_comparison(original_audio, stretched_audio, fs, speed_factor=1.5):
    """
    Plots the time-domain and frequency-domain (FFT) comparisons of the original and stretched audio signals.
    """
    # Time Axis for Original and Stretched Audio
    time_orig = np.arange(len(original_audio)) / fs
    time_stretched = np.arange(len(stretched_audio)) / fs

    # Compute FFT for Original Audio
    N_orig = len(original_audio)
    fft_orig = np.fft.fft(original_audio)
    fft_orig = fft_orig[:N_orig // 2]  # Take positive frequencies
    freq_orig = np.linspace(0, fs / 2, num=len(fft_orig))
    magnitude_orig = np.abs(fft_orig) / N_orig  # Normalize by number of samples

    # Compute FFT for Stretched Audio
    N_stretched = len(stretched_audio)
    fft_stretched = np.fft.fft(stretched_audio)
    fft_stretched = fft_stretched[:N_stretched // 2]  # Take positive frequencies
    freq_stretched = np.linspace(0, fs / 2, num=len(fft_stretched))
    magnitude_stretched = np.abs(fft_stretched) / N_stretched  # Normalize by number of samples

    plt.figure(figsize=(18, 12))

    # Time-Domain Plot: Original Audio
    plt.subplot(2, 2, 1)
    plt.plot(time_orig, original_audio, color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original Audio (Time Domain)")
    plt.grid(True)

    # Time-Domain Plot: Stretched Audio
    plt.subplot(2, 2, 2)
    plt.plot(time_stretched, stretched_audio, color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Stretched Audio (Time Domain) ({speed_factor}x Speed)")
    plt.grid(True)

    # Frequency-Domain Plot: Original Audio
    plt.subplot(2, 2, 3)
    plt.plot(freq_orig, magnitude_orig, color='blue')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.title("Original Audio (Frequency Domain)")
    plt.grid(True)
    plt.xlim(0, fs / 2)

    # Frequency-Domain Plot: Stretched Audio
    plt.subplot(2, 2, 4)
    plt.plot(freq_stretched, magnitude_stretched, color='orange')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.title(f"Stretched Audio (Frequency Domain) ({speed_factor}x Speed)")
    plt.grid(True)
    plt.xlim(0, fs / 2)

    plt.tight_layout()
    plt.show()


def save_plots(save_dir="./plots", filename=None):
    # Save the figure to the specified directory
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to release memory


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
    # Missing timeframes occur during unvoiced segments (like silence or noise) where no clear pitch is detected by the algorithm.
    # 3. plot_mel_spectrogram
    # 4. plot_energy_rms
    # 5. plot_audio_analysis
    plot_audio_analysis(resample_audio_16k, fs_16k, filename="resample_audio_16k.wav")
    plot_audio_analysis(resample_audio_manual, fs_16k, filename="resample_audio_manual.wav")
    # answers to the questions are in the analytical pdf

    # question 2(a), 2(b) - Add noise
    noise_audio = add_noise(resample_audio_16k)
    save_audio("noisy_speech.wav", fs_16k, noise_audio)

    # question 2(c) - plots
    plot_audio_analysis(resample_audio_16k, fs_16k, filename="resample_audio_16k.wav")
    fs_noise, stationary_noise = load_audio("stationary_noise.wav")
    plot_audio_analysis(stationary_noise, fs_noise, filename="stationary_noise.wav")
    plot_audio_analysis(noise_audio, fs_16k, filename="noisy_speech.wav")

    # question 3(a) - VAD
    enhanced_audio = spectral_subtraction(noise_audio)
    wavfile.write("enhanced.wav", fs_16k, enhanced_audio)
    # question 4(a) - AGC
    ths = 20 * np.log10(0.013)
    agc_audio, scaling_factors = auto_gain_control(resample_audio_16k, fs_16k, -20, ths)
    save_audio("agc_audio.wav", fs_16k, agc_audio)
    # question 4(b) - plots
    plot_audio_analysis(agc_audio, fs_16k, filename="agc_audio.wav")
    plot_results(resample_audio_16k, agc_audio, scaling_factors, fs_16k)

    # question 5 time-stretch the audio
    speed_factor = 1.5
    audio_stretched, magnitude_stretched, phase_stretched = time_stretch_phase_vocoder(resample_audio_16k, fs_16k,
                                                                                       speed_factor=speed_factor)
    stretched_filename = f'stretched_audio_{speed_factor}x.wav'
    save_audio(stretched_filename, fs_16k, audio_stretched)

    plot_audio_comparison(resample_audio_16k, audio_stretched, fs_16k, speed_factor)