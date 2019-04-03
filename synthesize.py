# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm

from matplotlib import pyplot as plt
from librosa import  display

def synthesize():
    # Load data
    L = load_data("synthesize")
    
    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2World')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))

        '''var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'WSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("WSRN Restored!")'''


        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.num_bap +hp.num_lf0+hp.num_mgc+hp.num_vuv), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.worlds: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]
            
        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        for i, world_tensor in enumerate(Y):
            print("Working on file", i+1,' world max,min:',world_tensor.max(),world_tensor.min())
            
            wav = world2wav(world_tensor)
            sf.write(hp.sampledir + "/{}.wav".format(i+1), wav,hp.sr_dataset)
        

        '''# Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})
        # Generate wav files
        if not os.path.exists(hp.sampledir+'_wsrn'): os.makedirs(hp.sampledir+'_wsrn')
        for i, world_tensor in enumerate(Y):
            print("Working on file", i+1)
            lf0,mgc,bap = tensor_to_world_features(world_tensor)
            wav = world2wav(lf0, mgc, bap,hp.frame_period_WSRN)
            sf.write(hp.sampledir+'_wsrn' + "/{}.wav".format(i+1), wav,hp.sr_dataset)'''
      

if __name__ == '__main__':
    synthesize()
    print("Done")


