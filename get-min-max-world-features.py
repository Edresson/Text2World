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

    
    
    

# Load data
fpaths, texts = load_data(mode="prepo") # list

if hp.phoneme == True:
    if hp.language =='pt':
        texts_to_phonemes(fpaths,texts)

lf0_max= 0
lf0_min = 999999999

mgc_max= 0
mgc_min = 999999999

bap_max= 0
bap_min = 999999999

for fpath in tqdm.tqdm(fpaths):
    try:
        if not os.path.exists("worlds"): os.mkdir("worlds")
        world=wav2world(os.path.join(hp.data_dir,fpath))
        lf0,mgc,bap = tensor_to_world_features(world)
        val = lf0.max()               
        if val > lf0_max:
            lf0_max = val
        val = lf0.min()                  
        if val < lf0_min:
            lf0_min = val

        val = mgc.max()                 
        if val > mgc_max:
            mgc_max = val
        val = mgc.min()                  
        if val < mgc_min:
            mgc_min = val

        val = bap.max()                
        if val > bap_max:
            bap_max = val
        val = bap.min()                  
        if val < bap_min:
            bap_min = val    
       
        '''num_padding = mel.shape[0]*8 - world.shape[0] 
        world = np.pad(world, [[0, num_padding], [0, 0]], mode="constant")'''
        #np.save("worlds/{}".format(data_list[i].replace("wav", "npy")), world
    except:
        pass


'''func= (0.9-0.1)*(value-min)/(max-min)+0.1
func_r = (func-0.1)/(0.9-0.1)*(max-min)+min'''


print("lf0:",lf0_min,';',lf0_max,"mgc:",mgc_min,';',mgc_max,"bap:",bap_min,';',bap_max)
