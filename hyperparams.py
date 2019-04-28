# -*- coding: utf-8 -*-
import os
class Hyperparams:
    '''Hyper parameters'''
    # pipeline
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.
    
    # signal processing
    sr = 22050  # Sampling rate.
    sr_dataset= 48000
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20

    #world features
    r = 4 # Reduction factor. Do not change this.
    num_mgc = 60
    num_lf0 = 1
    num_vuv = 1
    num_bap = 5 # 1(16KHz, 22050Hz), 5(44.1KHz, 48KHz)
    n_mgc = 60
    n_lf0 = 1
    n_vuv = 1
    n_bap = 5 # 1(16KHz, 22050Hz), 5(44.1KHz, 48KHz)
    
    frame_period = 12 #15ms
    sample_rate = 48000#22050 Hz (corresponding to ljspeech dataset)
    use_harvest = False 
    rescale_max = 0.999 #Rescaling value
    trim_silence = True #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    #frame_period = 40
    frame_period_WSRN = frame_period/r # 5
    
    f0_floor = 71.0
    f0_ceil = 800.0
    
    '''lf0_min= 0.0
    lf0_max= 6.67782
    mgc_min= -9.21034
    mgc_max=8.42621
    bap_min= -62.8978
    bap_max = 0'''

    # Model
    dropout_rate = 0.05
    vocoder = 'RTISI-LA' # or 
    #vocoder = 'griffin_lim'
    e = 128 # == embedding
    d = 256 # == hidden units of Text2Mel
    c = 512 # == hidden units of SSRN
    attention_win_size = 3

    # data
    data = "../TTS-Portuguese-Corpus"
    data_dir = os.path.join(data,'wavs/')
    # data = "/data/private/voice/kate"
    language = 'pt' # or 'eng'
    phoneme = False
    if phoneme == False and language == 'pt':
        test_data = 'phonetically-balanced-sentences.txt'
    elif phoneme == True and language == 'pt':
        test_data = 'phonetically-balanced-sentences-phoneme.txt'
    else:
        test_data = 'harvard_setences.txt'

    #vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding, E: EOS. #english
    vocab = "PE abcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû"#abcdefghijklmnopqrstuvwxyzçãõâôêíîúûñáéó.?" # P: Padding, E: EOS. #portuguese
    #portugues falta acento no a :"abcdefghijklmnopqrstuvwxyzçãõâôêíîúûñáéó.?"
    phoneme_vocab = "ˈoʧi.tulʊʤɪpaʒnsdk̃eɾvmzgɐ͂ɛxfbɣ,_ɔXqɲʃʎĩẽõhũŋcrɳ E"
    max_N = 180 # Maximum number of characters. default:180
    max_T = 3000 # Maximum number of world frames. default:210

    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "../logdir/LJ01"
    sampledir = '../logdirs-text2world/samples'
    B = 10 # batch size
    num_iterations =2000000
