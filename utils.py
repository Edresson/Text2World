# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function, division

import numpy as np
import librosa
import os, copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal
import codecs

import numpy as np
from numpy.lib.stride_tricks import as_strided
import copy
from scipy import fftpack

from hyperparams import Hyperparams as hp
import tensorflow as tf



import pysptk
import soundfile as sf
import pyworld as vocoder

int16_max = 32768.0
speed = hp.speed
frame_period = hp.frame_period
f0_floor = hp.f0_floor
f0_ceil = hp.f0_ceil

def normalize(x,max,min,min_inter=0.1,max_inter=0.9):
    return (max_inter-min_inter)*(x-min)/(max-min)+min_inter

    

def denormalize(x,max,min,min_inter=0.1,max_inter=0.9):
    return (x-min_inter)/(max_inter-min_inter)*(max-min)+min


def f0_normalize(x):
	return np.log(np.where(x == 0.0, 1.0, x)).astype(np.float32)

def f0_denormalize(x):
	return np.where(x == 0.0, 0.0, np.exp(x.astype(np.float64)))

def sp_normalize(x):
	sp = int16_max * np.sqrt(x)
	return pysptk.sptk.mcep(sp.astype(np.float32), order=hp.num_mgc - 1, alpha=hp.mcep_alpha,
				maxiter=0, threshold=0.001, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

def sp_denormalize(x):
	sp = pysptk.sptk.mgc2sp(x.astype(np.float64), order=hp.num_mgc - 1,
				alpha=hp.mcep_alpha, gamma=0.0, fftlen=hp.n_fft)
	return np.square(sp / int16_max)

def ap_normalize(x):
	return x.astype(np.float32)

def ap_denormalize(x, lf0):
	for i in range(len(lf0)):
		x[i] = np.where(lf0[i] == 0, np.zeros(x.shape[1]), x[i])
	return x.astype(np.float64)


def world_features_to_one_tensor(f0,sp,ap):
    return np.column_stack((np.column_stack((np.array(f0),np.array(sp))),np.array(ap)))

def tensor_to_world_features(tensor):
    f0=[]
    sp=[]
    ap = []
    sp_factor = hp.num_mgc+1
    for i in range(len(tensor)):
        f0.append(np.array(tensor[i][0]))
        sp.append(np.array(tensor[i][1:sp_factor]))
        ap.append(np.array(tensor[i][sp_factor:]))
        
    return np.array(f0),np.array(sp),np.array(ap)

def wav2world(wavfile):
    wav, fs = sf.read(wavfile)
    #f0,sp,ap=vocoder.wav2world(wav,fs , hp.n_fft, ap_depth=hp.num_bap)
    _f0_h, t_h = vocoder.dio(wav, fs, f0_floor=f0_floor, f0_ceil=f0_ceil,
                channels_in_octave=2,
                frame_period=frame_period,
                speed=speed)
    
    f0 = vocoder.stonemask(wav, _f0_h, t_h, fs)
    sp = vocoder.cheaptrick(wav, f0, t_h, fs)
    ap = vocoder.d4c(wav, f0, t_h, fs)
    # feature normalization
    lf0 = f0_normalize(f0)
    mgc = sp_normalize(sp)
    bap = ap_normalize(ap)
    f_normalize = np.vectorize(normalize)
    lf0= f_normalize(lf0,hp.lf0_max,hp.lf0_min)
    mgc= f_normalize(mgc,hp.mgc_max,hp.mgc_min)
    bap= f_normalize(bap,hp.bap_max,hp.bap_min)
    
    return np.array(world_features_to_one_tensor(lf0,mgc,bap))


def world2wav(lf0, mgc, bap):
	f_denormalize = np.vectorize(denormalize)
	lf0= f_denormalize(lf0,hp.lf0_max,hp.lf0_min)
	mgc= f_denormalize(mgc,hp.mgc_max,hp.mgc_min)
	bap= f_denormalize(bap,hp.bap_max,hp.bap_min)
	lf0 = np.where(lf0 < 1, 0.0, lf0)
	f0 = f0_denormalize(lf0)
	sp = sp_denormalize(mgc)
	ap = ap_denormalize(bap, lf0)
	print('features denomalize',lf0.shape,sp.shape,ap.shape)
	wav = vocoder.synthesize(f0, sp, ap,hp.sr_dataset,frame_period)
	return wav


def texts_to_phonemes(fpaths,texts,outputfile='texts-phoneme.csv',alphabet=False):
    from PETRUS.g2p.g2p import G2PTranscriber
    transcript = os.path.join(hp.data, outputfile)
    if alphabet == True:
        alphabet_list=[]
        alpha=os.path.join(hp.data, 'phoneme-alphabet.csv')

    transcript= codecs.open(transcript, 'w', 'utf-8')
    for i in range(len(texts)):
        words = texts[i].strip().lower().split(' ')
        transcrito = [] 
        for word in words:
            g2p = G2PTranscriber(word, algorithm='silva')
            transcription = g2p.transcriber()
            transcrito.append(transcription)
            if alphabet == True:
                for caracter in transcription:
                    if caracter not in alphabet_list:
                        alphabet_list.append(caracter)

        frase = str(fpaths[i])+'=='+"_".join(transcrito)+'\n'
        transcript.write(frase)
    if alphabet == True:
        alphabet = codecs.open(alpha, 'w', 'utf-8')
        for i in alphabet_list:
            alphabet.write(i)


def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.

    Args:
      fpath: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
     '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
 
    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)

    return mel


def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram

    Args:
      mag: A numpy array of (T, 1+n_fft//2)

    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
    # to amplitude
    mag = np.power(10.0, mag * 0.05)
    if hp.vocoder == 'griffin_lim':
        # wav reconstruction
        wav = griffin_lim(mag**hp.power)
    elif hp.vocoder == 'RTISI-LA' : # RTISI-LA
        wav = iterate_invert_spectrogram(mag**hp.power)
        
    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def iterate_invert_spectrogram(X_s, n_iter=10, verbose=False,
                               complex_input=False):
    """
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    for i in range(n_iter):
        if verbose:
            print("Runnning iter %i" % i)
        if i == 0 and not complex_input:
            X_t = invert_spectrogram(X_best)
        else:
            # Calculate offset was False in the MATLAB version
            # but in mine it massively improves the result
            # Possible bug in my impl?
            X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(reg, np.abs(est))
        phase = phase[:len(X_s)]
        X_s = X_s[:len(phase)]
        X_best = X_s * phase
    X_t = invert_spectrogram(X_best)
    return np.real(X_t)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def plot_alignment(alignment, gs, dir=hp.logdir):
    """Plots the alignment.

    Args:
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      gs: (int) global step.
      dir: Output path.
    """
    if not os.path.exists(dir): os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}.png'.format(dir, gs), format='png')

def guided_attention(g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    W = np.zeros((hp.max_N, hp.max_T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(hp.max_T) - n_pos / float(hp.max_N)) ** 2 / (2 * g * g))
    return W

def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

def load_spectrograms(fpath):
    '''Read the wave file in `fpath`
    and extracts spectrograms'''

    fname = os.path.basename(fpath)
    mel = get_spectrograms(fpath)
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    

    # Reduction
    mel = mel[::hp.r, :]
    return fname, mel

