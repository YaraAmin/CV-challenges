#importing libraries
! pip install imgaug
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import math
import sys
from imgaug import augmenters as aug
import random
from datetime import datetime
import os
import seaborn as sns
#loading data
os.chdir('D:/CS/computervision/New folder/cifar-100-python')
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict
meta = unpickle('meta')
train = unpickle('train')
test = unpickle('test')
Classes = pd.DataFrame(meta[b'fine_label_names'],columns = ['Classes'])
X = train[b"data"]
#reshape X 
X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
#apply Augmentation techniques to slightly change the data and fool the model
Au = aug.Sequential([
    aug.Fliplr(0.5),
    aug.CropAndPad(px=(-2, 2),sample_independently=True,pad_mode=["constant", "edge"]),
    aug.Affine(shear=(-10, 10),mode = ['symmetric','wrap']),
    aug.Add((-5, 5)),
    aug.Multiply((0.8, 1.2)),

],random_order=True)
    
#make 10 folds
X1 = Au.augment_images(X)
X2 = Au.augment_images(X)
X3 = Au.augment_images(X)
X4 = Au.augment_images(X)
X5 = Au.augment_images(X)
X6 = Au.augment_images(X)
X7 = Au.augment_images(X)
X8 = Au.augment_images(X)
X9 = Au.augment_images(X)
X10 = Au.augment_images(X)
#encode the categorical labels
def one_hot_encode(vec, vals=num_class):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out
#divide by 255 to find the minima
train = []
train.extend(X/255)
train.extend(X1/255)
train.extend(X2/255)
train.extend(X3/255)
train.extend(X4/255)
train.extend(X5/255)
train.extend(X6/255)
train.extend(X7/255)
train.extend(X8/255)
train.extend(X9/255)
train.extend(X10/255)
#apply on test data
labels = []
labels.extend(train[b'fine_labels'])
labels.extend(train[b'fine_labels'])
labels.extend(train[b'fine_labels'])
labels.extend(train[b'fine_labels'])
labels.extend(train[b'fine_labels'])
labels.extend(train[b'fine_labels'])
labels.extend(train[b'fine_labels'])
labels.extend(train[b'fine_labels'])
labels.extend(train[b'fine_labels'])
labels.extend(train[b'fine_labels'])
#shuffle the data to random any pattern in the order
train_shuffled = []
labels_shuffled = []

combined = list(zip(train, labels))
random.shuffle(combined)

train_shuffled[:], labels_shuffled[:] = zip(*combined)
train_shuffled = np.asarray(train_shuffled)
num_of_class = 100
train_len = len(train_shuffled)
#encode train labels
labels_shuffled= one_hot_encode(labels_shuffled, num_of_class)
test_shuffled = np.vstack(test[b"data"])
test_len = len(test_shuffled)
test_shuffled = test_shuffled.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
#encode test labels
test_labels = one_hot_encode(test[b'fine_labels'], num_of_class)
class CifarHelper():
    
    def __init__(self):
        self.i = 0
        
        self.training_images = train_shuffled
        self.training_labels = labels_shuffled
        
        self.test_images = test_shuffled
        self.test_labels = test_labels
        
    def next_batch(self, batch_size=mini_batch_size):
        
        x = self.training_images[self.i:self.i+batch_size].reshape(100,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y
ch = CifarHelper()
#tensorflow
x = tf.placeholder(tf.float32,shape=[None,32,32,3])
y = tf.placeholder(tf.float32,shape=[None,num_class])
hold_prob = tf.placeholder(tf.float32)
#functions for initializing layers
def weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_size2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
num_class = 100
keep_prob =0.5
#create the model
#layer1
convo_1 = convolutional_layer(x,shape=[5,5,3,32])
convo_1_pooling = max_pool_size2(convo_1)
convo_1_bn = tf.layers.batch_normalization(convo_1_pooling)
convo_1_dp = tf.nn.dropout(convo_1_bn, keep_prob)

#layer2
convo_2 = convolutional_layer(convo_1_dp,shape=[5,5,32,64])
convo_2_pooling = max_pool_size2(convo_2)
convo_2_bn = tf.layers.batch_normalization(convo_2_pooling)
convo_2_dp = tf.nn.dropout(convo_2_bn, keep_prob)

#layer3
convo_3 = convolutional_layer(convo_2_pooling,shape=[3,3,64,128])
convo_3_pooling = max_pool_size2(convo_3)
convo_3_bn = tf.layers.batch_normalization(convo_3_pooling)
convo_3_dp = tf.nn.dropout(convo_3_bn, keep_prob)

#layer4
convo_4 = convolutional_layer(convo_3,shape=[1,1,128,256])
convo_4_pooling = max_pool_size2(convo_4)
convo_4_bn = tf.layers.batch_normalization(convo_4_pooling)
convo_4_dp = tf.nn.dropout(convo_4_bn, keep_prob)

convo_flat = tf.reshape(convo_4_pooling,[-1,8*8*256])
full_layer_one = tf.nn.relu(normal_full_layer(convo_flat,1024))
full_one_dropout = tf.nn.dropout(full_layer_one,hold_prob)
y_pred = normal_full_layer(full_one_dropout,num_class)

#calc loss function
soft_max = tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = y_pred)
cross_entropy = tf.reduce_mean(soft_max)
#Adam optimizer
optimizer = tf.train.AdamOptimizer(.001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer() #for gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()
#run the model
%%time
epoch = 200000
minibatch_check = 500
accuracy_list = []
accuracy = 0
target_accuracy  = .55
with tf.Session(config = config) as sess:
    sess.run(init)
    i = 0
    while (accuracy<target_accuracy):
        i = i+1
        
        batch = ch.next_batch(100)
        
        sess.run(train,feed_dict = {x:batch[0],y_true:batch[1],hold_prob:0.5})
        
        if i % 400 == 0:
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            
            print('Train Accuracy:')
            print(sess.run(acc,feed_dict = {x:batch[0],y_true:batch[1],hold_prob:1.0}))
            batch_accuracy = []
            
            for k in range(0,int(len(test_shuffled)/minibatch_check)):
                batch_accuracy.append(sess.run(acc,feed_dict={x:test_shuffled[minibatch_check*(k):minibatch_check*(k+1)],
                                                              y:test_labels[minibatch_check*(k):minibatch_check*(k+1)],
                                                              hold_prob:1.0}))
            print('TEST ACCURACY:')   
            accuracy = sum(batch_accuracy) / (len(batch_accuracy))
            print(accuracy)
            accuracy_list.append(accuracy)
            print('\n') 

             
            
        if (accuracy>target_accuracy):
            saver.save(sess,'D:/CS/computervision/New folder')
            
        plt.plot(accuracy_list)
#predictions
predictions_df = np.argmax(predictions,1)
predictions_df = pd.DataFrame(predictions_df)

test_labels_df = np.argmax(test_labels,1)
test_labels_df = pd.DataFrame(test_labels_df)
Classes = pd.DataFrame(Classes)

#function that takes an image as input as predict its classification
model_path = 'models/model53.ckpt'
#input the image path
os.getcwd('')
def image_prediction(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(img, (32, 32)) 
    try_out = []
    try_out.append(resized_image/255)

    
    predictions = []
    with tf.Session() as sess:

        saver.restore(sess,model_path)

        probabilities = tf.nn.softmax(y_pred)
        matches2 = softmaxx
        acc2 = tf.cast(probabilities,tf.float32)

        predictions.extend(sess.run(acc2,feed_dict={x:try_out,hold_prob:1.0}))
        predictions = np.array(predictions)

        predictions_df = pd.DataFrame(predictions).T
        predictions_df = predictions_df.sort_values(0,ascending=0)
        predictions_df = predictions_df[:10].T
        predictions_df.columns = Classes.iloc[predictions_df.columns.values]


        columns = predictions_df.columns
        columns_list = []
        for i in range(len(columns)):
            columns_list.append(str(columns[i])[3:-3])
                columns_list = pd.DataFrame(columns_list)
        predictions_df.columns = pd.DataFrame(columns_list)

        predictions_df = predictions_df.T
        predictions_df.columns=['Probability']
        predictions_df['Prediction'] = predictions_df.index
        
        f, axarr = plt.subplots(1,2, figsize=(10,4))

        axarr[0].imshow(img)
        axarr[0].axis('off')

        axarr[1] = sns.barplot(x="Probability", y="Prediction", data=predictions_df,color="red",)
        sns.set_style(style='white')

        axarr[1].set_ylabel('')    
        axarr[1].set_xlabel('')
        axarr[1].grid(False)
        axarr[1].spines["top"].set_visible(False)
        axarr[1].spines["right"].set_visible(False)
        axarr[1].spines["bottom"].set_visible(False)
        axarr[1].spines["left"].set_visible(False)
        
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        f.suptitle("Model Prediction")
        f.subplots_adjust(top=0.88)
