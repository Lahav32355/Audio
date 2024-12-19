import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy.signal import resample
import librosa
import librosa.display
import librosa.feature
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


def add_noise(audio):
    noise_fs, noise = load_audio("stationary_noise.wav")
    resampled_noise, _ = resample_audio(noise, noise_fs, 16000)
    if len(resampled_noise) < len(audio):
        return audio[:len(resampled_noise)] + resampled_noise
    return audio + resampled_noise[:len(audio)]  # Add noise to the audio



#Todo:checking the order
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




