# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from utils import load_spectrograms
import os
from data_load import load_data
import numpy as np
import tqdm
from hyperparams import Hyperparams as hp
import codecs
from utils import *

def texts_to_phonemes(fpaths,texts):
    from PETRUS.g2p.g2p import G2PTranscriber
    transcript = os.path.join(hp.data, 'texts-phoneme.csv')
    alpha=os.path.join(hp.data, 'phoneme-alphabet.csv')
    transcript= codecs.open(transcript, 'w', 'utf-8')
    alphabet_list=[]
    #print('Texts:',texts)
    for i in range(len(texts)):
        texts[i]=texts[i].replace(',',' , ').replace('?',' ? ')
        words = texts[i].strip().lower().split(' ')
        transcrito = [] 
        for word in words:
            #print(word)
            # Initialize g2p transcriber
            g2p = G2PTranscriber(word, algorithm='silva')
            transcription = g2p.transcriber()
            transcrito.append(transcription)
            for caracter in transcription:
                if caracter not in alphabet_list:
                    alphabet_list.append(caracter)
           
        #print('Frase: ',"_".join(words))
        #print('Transcricao: ',"_".join(transcrito))

        frase = str(fpaths[i].replace(hp.data,''))+'=='+"_".join(transcrito)+'\n'
        transcript.write(frase)

    alphabet = codecs.open(alpha, 'w', 'utf-8')
    print('Alfabeto:',alphabet_list)
    for i in alphabet_list:
        alphabet.write(i)
    
    
    

# Load data
fpaths, texts = load_data(mode="prepo") # list

if hp.phoneme == True:
    if hp.language =='pt':
        texts_to_phonemes(fpaths,texts)

for fpath in tqdm.tqdm(fpaths):
    try:    
        if not os.path.exists("worlds"): os.mkdir("worlds")
        if not os.path.exists("worlds_wsr"): os.mkdir("worlds_wsr")
        #world=wav2world(fpath,hp.frame_period)
        world_wsr=wav2world(fpath,hp.frame_period_WSRN)
        #np.save("worlds/{}".format( os.path.basename(fpath).replace("wav", "npy")), world)
        np.save("worlds_wsr/{}".format( os.path.basename(fpath).replace("wav", "npy")), world_wsr)
    except:    
        print('file: ',fpath,' ignored')
        continue
