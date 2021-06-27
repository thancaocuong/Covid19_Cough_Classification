import librosa
import numpy as np

def width_padding(array, desired_w):
    w = array.shape[1]
    if w > desired_w:
        return array[:, :desired_w]
    else:
        b = (desired_w - w) // 2
        bb = desired_w - b - w
        return np.pad(array, pad_width=((0, 0), (b, bb)), mode='constant')

def extract_feature(audio, fs, segment_size_t=0.025, n_mfcc=26, n_fft=256, hop_length=40, audio_transfroms=None):
    if audio_transfroms is not None:
        audio, fs = audio_transfroms(audio, fs)
    mfcc_feature = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc_feature)
    mfcc_delta2 = librosa.feature.delta(mfcc_delta, order=2)
    # zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft, hop_length=hop_length)
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length= hop_length)
    chroma_stft = librosa.feature.chroma_stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    return np.concatenate([mfcc_feature, mfcc_delta, mfcc_delta2, zcr, stft, chroma_stft])


def mfcc_feature(audio, fs, audio_transforms):
    segment_size_t=0.025
    n_mfcc=39
    n_fft=256
    num_seg = 256
    hop_length=len(audio)//num_seg

    feature = extract_feature(audio,
                             fs,
                             segment_size_t,
                             n_mfcc,
                             n_fft,
                             hop_length,
                             audio_transfroms=audio_transforms)
    # padding or trucate to the same width
    feature = width_padding(feature, num_seg)
    return feature[None, ...].astype(np.float64)