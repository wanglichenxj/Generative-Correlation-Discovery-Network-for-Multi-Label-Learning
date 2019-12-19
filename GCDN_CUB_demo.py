# =====================
# Generative Correlation Discovery Network for Multi-Label Learning
# =====================
# Author: Lichen Wang
# Date: Nov., 2019
# E-mail: wanglichenxj@gmail.com

# @inproceedings{GCDN_ICDM19_Lichen,
#   title={Generative Correlation Discovery Network for Multi-Label Learning},
#   author={Wang, Lichen and Ding, Zhengming and Han, Seungju and Han, Jae-Joon and Choi, Changkyu and Fu, Yun},
#   booktitle={Proceedings of IEEE International Conference on Data Mining},
#   year={2019}
# }
# =====================

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy.io # Load and write MATLAB mat file
import random # Sample the label
import evaluation_GCDN as evaluation

# Set GPU number used to run
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ==============================
# Load and arrange the original data
# ==============================
print ('Load data ...')
mat = scipy.io.loadmat('datasets/cub_data_VGG_ori.mat')
Xl=mat['Xl']
Xt=mat['Xu']
Yl=mat['Sl']
Yt=mat['Su']
feature_dim=len(Xl) # Get featrue dimension
label_dim=len(Yl) # Get label dimension
data_number_labeled=len(Xl[0]) # Get the number of training/labeled data
data_number_test=len(Xt[0]) # Get the number of testing data
Xl=Xl.T # Transpose data
Xt=Xt.T
Yl=Yl.T
Yt=Yt.T

# ==============================
# Set parameters
# ==============================
Z_dim=100 # Noise dimension
mb_size=64 # Batch size
h_dim=200 # Hidden layer dimensions
epsilon = 1e-6

# ====================
# Define random matrix for weights
# ====================
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# ====================
# Generate random noise for conditional GAN
# ====================
def sample_Z(m, n):
    return np.random.uniform(0., 1., size=[m, n])

# ====================
# Define the leaky ReLU activation
# ====================
def leak_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

print ('Construct network...')
# ====================
# Define placeholders
# ====================
X = tf.placeholder(tf.float32, shape=[None, feature_dim]) # Input feature data
y = tf.placeholder(tf.float32, shape=[None, label_dim]) # Input label data
Z = tf.placeholder(tf.float32, shape=[None, Z_dim]) # Input noise data

# ====================
# Define discriminator
# ====================
D_W1 = tf.Variable(xavier_init([feature_dim + label_dim, h_dim]),name='D_W1') # 1-layer
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]),name='D_b1')
D_W2 = tf.Variable(xavier_init([h_dim, h_dim]),name='D_W2') # 2-layer
D_b2 = tf.Variable(tf.zeros(shape=[h_dim]),name='D_b2')
D_M1 = tf.Variable(xavier_init([h_dim, 15]),name='D_M1') # define mini-batch
D_Mb1 = tf.Variable(xavier_init([15]),name='D_Mb1')
D_W3 = tf.Variable(xavier_init([205, 1]),name='D_W3') # 3-layer
D_b3 = tf.Variable(tf.zeros(shape=[1]),name='D_b3')
theta_D = [D_W1, D_W2, D_b1, D_b2, D_W3, D_b3,D_M1,D_Mb1]

# Define mini-batch
def minibatch(input, num_kernels=5, kernel_dim=3):
    x = tf.matmul(input, D_M1) + D_Mb1
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2) # eps
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat(axis=1, values=[input, minibatch_features])

# Define discriminator
def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])    
    D_h1 = leak_relu(tf.matmul(inputs, D_W1) + D_b1,0.2)
    D_h2 = leak_relu(tf.matmul(D_h1, D_W2) + D_b2,0.2)
    D_h3 = minibatch(D_h2)
    D_h3 = tf.matmul(D_h3, D_W3) + D_b3
    D_logit = D_h3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


# ====================
# Define generator
# ====================
G_w1_BN = tf.Variable(xavier_init([Z_dim + label_dim, h_dim]),name='G_W1') # 1-layer
G_scale1 = tf.Variable(tf.ones([h_dim]))
G_beta1 = tf.Variable(tf.zeros([h_dim]))
G_w2_BN = tf.Variable(xavier_init([h_dim, feature_dim]),name='G_W2') # 2-layer
G_scale2 = tf.Variable(tf.ones([feature_dim]))
G_beta2 = tf.Variable(tf.zeros([feature_dim]))
theta_G = [G_w1_BN, G_scale1, G_beta1, G_w2_BN, G_scale2, G_beta2]

# Define generator
def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    # 1st layer batch nornalization
    G_z1_BN = tf.matmul(inputs,G_w1_BN)
    G_batch_mean1, G_batch_var1 = tf.nn.moments(G_z1_BN,[0])
    G_z1_hat = (G_z1_BN - G_batch_mean1) / tf.sqrt(G_batch_var1 + epsilon)
    G_BN1 = G_scale1 * G_z1_hat + G_beta1
    G_l1_BN = tf.nn.relu(G_BN1)
    # 2nd layer + bath normalization
    G_z2_BN = tf.matmul(G_l1_BN,G_w2_BN)
    G_batch_mean2, G_batch_var2 = tf.nn.moments(G_z2_BN,[0])
    G_BN2 = tf.nn.batch_normalization(G_z2_BN,G_batch_mean2,G_batch_var2,G_beta2,G_scale2,epsilon)
    G_l2_BN = G_BN2
    G_prob = tf.nn.relu(G_l2_BN)
    return G_prob


# ====================
# Define classifier and Correlation Discovery Network (CDN)
# ====================
hh_dim = 2000
C_W1 = tf.Variable(xavier_init([feature_dim, h_dim]))
C_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
C_W2 = tf.Variable(xavier_init([h_dim, label_dim]))
C_b2 = tf.Variable(tf.zeros(shape=[label_dim]))

C_W3 = tf.Variable(xavier_init([label_dim*label_dim, hh_dim]))
C_b3 = tf.Variable(tf.zeros(shape=[hh_dim]))
C_W4 = tf.Variable(xavier_init([hh_dim, label_dim]))
C_b4 = tf.Variable(tf.zeros(shape=[label_dim]))
theta_C = [C_W1, C_W2, C_W3, C_W4, C_b1, C_b2, C_b3, C_b4]

# Define the C_M(.) and C_CDN(.)
def classifier(X):
    # regular multi-label classifier C_M(.)
    inputs = X
    C_h1 = tf.nn.relu(tf.matmul(inputs, C_W1) + C_b1)
    C_log_prob = tf.matmul(C_h1, C_W2) + C_b2
    C_prob = tf.nn.sigmoid(C_log_prob) # initial prediction from C_M(.)
    
    # Define Correlation Discovery Network C_CDN(.)
    C_prob_1 = tf.expand_dims(C_prob, -1)
    C_prob_2 = tf.expand_dims(C_prob, 1)
    W_feature = tf.matmul(C_prob_1, C_prob_2)
    W_feature_1 = tf.reshape(W_feature, [-1, label_dim*label_dim])
    C_h2 = tf.nn.relu(tf.matmul(W_feature_1, C_W3) + C_b3)
    C_feature = tf.nn.sigmoid(tf.matmul(C_h2, C_W4) + C_b4) # final output from C_CDN(.)
    return C_prob, C_feature


# ===============
# Define output of generator, classifier, and discriminator 
# ===============
G_sample = generator(Z, y) # Input noise and label
classified_label_G_prob, classified_label_G_graph = classifier(G_sample) # Classify the generated features (non-graph + graph)
classified_label_X_prob, classified_label_X_graph = classifier(X) # Classify the ground truth features (non-graph + graph)
D_real, D_logit_real = discriminator(X, y) # Input ground truth feature and label
D_fake, D_logit_fake = discriminator(G_sample, y) # Input generated feature and label

# ================
# define loss function and solver
# ================
# Descriminator loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
D_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(D_loss, var_list=theta_D)

# Generator loss
G_loss_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
G_loss_fea = tf.reduce_mean(tf.square(G_sample-X)) # sample similarity ||generated_sample - real_sample||
G_loss_sum = G_loss_dis + 10*G_loss_fea
G_solver = tf.train.AdamOptimizer(learning_rate=0.003).minimize(G_loss_sum, var_list=theta_G)

# Pre-trained generator loss
p_loss = tf.reduce_mean(tf.square(X-G_sample))
p_solver = tf.train.AdamOptimizer().minimize(p_loss, var_list=theta_G)

# Classification loss of C_M(.)
GGG = 1
C_loss_cls_G_prob = tf.reduce_mean(tf.square(y-classified_label_G_prob)) # C loss for generated feature to label
C_loss_cls_X_prob = tf.reduce_mean(tf.square(y-classified_label_X_prob)) # C loss for generated feature to label
C_loss_sum_prob = C_loss_cls_X_prob + GGG*C_loss_cls_G_prob # Consider both real and fake features

# Classification loss of C_CDN(.)
C_loss_cls_G_graph = tf.reduce_mean(tf.square(y-classified_label_G_graph))
C_loss_cls_X_graph = tf.reduce_mean(tf.square(y-classified_label_X_graph))
C_loss_sum_graph = C_loss_cls_X_graph + GGG*C_loss_cls_G_graph
lbd = 0.5
C_loss = lbd*C_loss_sum_prob + (1-lbd)*C_loss_sum_graph
C_solver_sum = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(C_loss, var_list=theta_C)

# ================
# Start training
# ================
# Initialization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

print('Start training ... ')

for seperate_train in range(300):
    for it in range(100):
        rand_idx = random.sample(range(data_number_labeled),mb_size) # get the index of the label
        X_mb = Xl[rand_idx,:] # random select the training sample
        y_mb = Yl[rand_idx,:]
        Z_sample = 10*sample_Z(mb_size, Z_dim)

        # Train discriminator
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})

        # Train generator
        for upG in range(8):
            _, G_loss_curr = sess.run([G_solver, G_loss_sum], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
        
        # Train classifier
        for upC in range(1):
            _, C_loss_curr = sess.run([C_solver_sum, C_loss], feed_dict={X:X_mb, Z:Z_sample, y:y_mb})
    
    # Show testing performance
    if seperate_train % 1 == 0:
        cls_label_Ft = sess.run(classified_label_X_graph, feed_dict={X:Xt})      
        # =======================
        # Evaluation
        # Note: mAP evaluation is time consuming. Set get_mAP=True/False to active/deactivate the mAP evaluation
        #       If mAP is deactivated, the mAP would be 0 in the output
        # =======================
        prec, rec, f1, retrieved, f1Ind, precInd, recInd, _, mAP = evaluation.eva(Yt.T, cls_label_Ft.T, 5, get_mAP=False)
        print('loop=',seperate_train,'  Prec = {:.4f}'.format(prec), '  Rec = {:.4f}'.format(rec), '  F1 = {:.4f}'.format(f1), '  N-R = ', retrieved, '  mAP = {:.4f}'.format(mAP))
