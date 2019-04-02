# -*- coding: utf-8 -*-
# /usr/bin/python2
from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention,WSRN
import tensorflow as tf
from utils import *
import sys


class Graph:
    def __init__(self, num=1,mode="train"):
        '''
        Args:
          mode: Either "train" or "synthesize".
        '''
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set flag
        training = True if mode=="train" else False

        # Graph
        # Data Feeding
        ## L: Text. (B, N), int32
        ## world: World Vocoder concatenate tensor.(B, 8*T/r, num_lf0+num_mgc+num_bap) float32
        if mode=="train":
            self.L, self.worlds,self.worlds_WSR, self.fnames, self.num_batch = get_batch()
            self.prev_max_attentions = tf.ones(shape=(hp.B,), dtype=tf.int32)
            self.gts = tf.convert_to_tensor(guided_attention())
        else:  # Synthesize
            self.L = tf.placeholder(tf.int32, shape=(None, None))
            self.worlds = tf.placeholder(tf.float32, shape=(None, None, hp.num_bap+hp.num_lf0+hp.num_mgc))
            self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))
            self.gts = tf.convert_to_tensor(guided_attention())

        if num==1 or (not training):
            with tf.variable_scope("Text2World"):
                # Get S or decoder inputs. (B, 8*T/r, num_lf0+num_mgc+num_bap)
                self.S = tf.concat((tf.zeros_like(self.worlds[:, :1, :]), self.worlds[:, :-1, :]), 1)

                # Networks
                with tf.variable_scope("TextEnc"):
                    self.K, self.V = TextEnc(self.L, training=training)  # (N, Tx, e)

                with tf.variable_scope("AudioEnc"):
                    self.Q = AudioEnc(self.S, training=training)

                with tf.variable_scope("Attention"):
                    # R: (B, T/r, 2d)
                    # alignments: (B, N, T/r)
                    # max_attentions: (B,)
                    self.R, self.alignments, self.max_attentions = Attention(self.Q, self.K, self.V,
                                                                             mononotic_attention=(not training),
                                                                             prev_max_attentions=self.prev_max_attentions)
                with tf.variable_scope("AudioDec"):
                    self.Y_logits, self.Y = AudioDec(self.R, training=training) # (B, T/r, num_lf0+num_mgc+num_bap)
        else:  # num==2 & training. Note that during training,
            with tf.variable_scope("WSRN"):
                self.Z_logits, self.Z = WSRN(self.worlds, training=training)

        if not training:
            # During inference, the predicted melspectrogram values are fed.
            with tf.variable_scope("WSRN"):
                self.Z_logits, self.Z = WSRN(self.Y, training=training)

        with tf.variable_scope("gs"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)



        if training:
            if num==1: # Text2World
                # world L1 loss
                self.loss_worlds = tf.losses.mean_squared_error(self.Y,self.worlds)
                #self.loss_worlds = tf.reduce_mean(tf.abs(self.Y - self.worlds))

                # world binary divergence loss
                #self.loss_bd1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Y_logits, labels=self.worlds))

                # guided_attention loss
                self.A = tf.pad(self.alignments, [(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT", constant_values=-1.)[:, :hp.max_N, :hp.max_T]
                self.attention_masks = tf.to_float(tf.not_equal(self.A, -1))
                self.loss_att = tf.reduce_sum(tf.abs(self.A * self.gts) * self.attention_masks)
                self.mask_sum = tf.reduce_sum(self.attention_masks)
                self.loss_att /= self.mask_sum

                # total loss
                self.loss = self.loss_worlds #+  self.loss_att#self.loss_bd1 +

                tf.summary.scalar('train/loss_worlds', self.loss_worlds)
                #tf.summary.scalar('train/loss_bd1', self.loss_bd1)
                tf.summary.scalar('train/loss_att', self.loss_att)
                tf.summary.image('train/world_gt', tf.expand_dims(tf.transpose(self.worlds[:1], [0, 2, 1]), -1))
                tf.summary.image('train/world_hat', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))
            else:#WSRN
                # world L1 loss
                self.loss_WSR = tf.reduce_mean(tf.abs(self.Z - self.worlds_WSR))

                # world binary divergence loss
                #self.loss_bd2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Z_logits, labels=self.worlds_WSR))

                # total loss
                self.loss = self.loss_WSR #+ self.loss_bd2

                tf.summary.scalar('train/loss_mags', self.loss_WSR)
                #tf.summary.scalar('train/loss_bd2', self.loss_bd2)

            # Training Scheme
            self.lr = learning_rate_decay(hp.lr, self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            tf.summary.scalar("lr", self.lr)

            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_value(grad, -1., 1.)
                self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            # Summary
            self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    num = int(sys.argv[1])

    g = Graph(num=num);print("Training Graph loaded")

    logdir = hp.logdir + "-" + str(num)
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)
    sv.saver._max_to_keep=1000 # set max number checkpoint is save

    with sv.managed_session() as sess:
        while 1:
            for _ in range(g.num_batch):
                gs,loss, _ = sess.run([g.global_step,g.loss, g.train_op])

                # Write checkpoint files at every 1k steps
                if gs % 1000 == 0:
                    sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))
                    print("processados "+str(gs)+" de "+str(hp.num_iterations), ' loss: ',loss)

                    if num==1:
                        # plot alignment
                        alignments = sess.run(g.alignments)
                        plot_alignment(alignments[0], str(gs // 1000).zfill(3) + "k", logdir)

                # break
                if gs > hp.num_iterations: 
                    break

    print("Done")
