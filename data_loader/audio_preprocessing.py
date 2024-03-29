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

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y

def trim_and_pad(audio, max_samples):
    audio_length = audio.shape[0]
    if audio_length > max_samples:
        # trim long_data
        trim_length = audio_length - max_samples
        audio = audio[int(trim_length//2):int(max_samples+trim_length//2)]
    else:
        # n_repeats = max_samples // len(audio)
        # epsilon = max_samples % len(audio)
        
        # audio = np.concatenate([audio]*n_repeats + [audio[:epsilon]])
        padding = int(max_samples - audio_length)
        offset = int(padding // 2)
        audio = np.pad(audio, (offset, max_samples - audio_length - offset), 'constant')
    
    return audio


def segments(audio, fs, segment_size_t=0.05):
    audio_len = len(audio)
    segment_size = int(segment_size_t * fs)  # segment size in samples
    # Break signal into list of segments in a single-line Python code
    segments = np.array([audio[x:x + segment_size] for x in
                         np.arange(0, audio_len, segment_size)])
    return segments

def remove_silent(audio, fs, segment_size_t, v2=False):
    normalized_segments = segments(audio, fs, segment_size_t)
    energies = np.array([(s**2).sum() / len(s) for s in normalized_segments])
    threshold = 0.4 * np.median(energies)
    index_of_segments_to_keep = (np.where(energies > threshold)[0])
    # get segments that have energies higher than a the threshold:
    high_energy_segments = normalized_segments[index_of_segments_to_keep]
    try:
        return np.concatenate(high_energy_segments)
    except:
        return audio

def extract_mfcc_feature(audio, fs, mfcc_config, audio_transforms=None, for_test=False):
    # n_mfcc=15
    # n_fft=1024
    # hop_length= 256
    # max_samples = int(7.5 * 8000) # 7.5s

    # do_remove_silent = mfcc_config.get("do_remove_silent", False)
    n_mfcc = mfcc_config.get("n_mfcc", 15)
    n_fft = mfcc_config.get("n_fft", 1024)
    hop_length = mfcc_config.get("hop_length", 256)
    max_duration = mfcc_config.get("max_duration", 15)
    target_sr = mfcc_config.get("target_sr", 48000)
    max_samples = int(max_duration * target_sr)
    if for_test:
        # if it's the test set -> do remove silent and  resample
        audio = remove_silent(audio, fs, segment_size_t=0.025)
        audio = librosa.resample(audio, fs, target_sr)
        fs = target_sr
    if audio_transforms is not None:
        try:
            audio, fs = audio_transforms(audio, fs)
        except:
            audio = audio_transforms(samples=audio, sample_rate=fs)
    audio = trim_and_pad(audio, max_samples)
    mfcc_feature = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc_feature[None, ...].astype(np.float64)

def extract_feature(audio, fs, segment_size_t=0.025, n_mfcc=26, n_fft=256, hop_length=40, audio_transfroms=None):
    # audio = remove_silent(audio, fs, segment_size_t)
    if audio_transfroms is not None:
        try:
            audio, fs = audio_transfroms(audio, fs)
        except:
            audio = audio_transfroms(samples=audio, sample_rate=fs)
    mfcc_feature = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc_feature)
    mfcc_delta2 = librosa.feature.delta(mfcc_delta, order=2)
    # zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft, hop_length=hop_length)
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length= hop_length)
    chroma_stft = librosa.feature.chroma_stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    return np.concatenate([mfcc_feature, mfcc_delta, mfcc_delta2, zcr, stft, chroma_stft])


def mfcc_feature(audio, fs, audio_transforms=None):
    segment_size_t=0.025
    n_mfcc=39
    n_fft=256
    num_seg = 128
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