import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
import pickle
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import keras
import math
import sys

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(confusion_matrix(y_true, y_pred))

# Training Settings
batch_size = 100
latent_dim = 800
change = 10
units = 800  # num unit in the MLP hidden layer
num_filter_ae_cls = [32, 32, 64, 64, 128, 128]  # conv_layers and No. of its channels for AE + CLS
num_filter_cls = []  # conv_layers and No. of its channel for only cls
num_dense = 0  # number of dense layer in classifier excluding the last layer
kernel_size = (1, 3)
activation = tf.nn.relu
padding = 'same'
strides = 1
pool_size = (1, 2)
num_class = 5
reg_l2 = tf.contrib.layers.l1_regularizer(scale=0.1)
initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
#initializer = tf.truncated_normal_initializer()

# Import the data
#filename = '../Mode-codes-Revised/paper2_data_for_DL_train_val_test.pickle'
filename = '/home/sxz/data/geolife_Data/paper2_data_for_DL_kfold_dataset_RL.pickle'
encode_len = 0
with open(filename, 'rb') as f:
    kfold_dataset, X_unlabeled = pickle.load(f)
    for i in range(5):
        print(np.shape(kfold_dataset[i][3]))
        encode_len += len(kfold_dataset[i][3])
        # 统计kfold_dataset的大小
    print(encode_len)
    # print(len(kfold_dataset[2][2])
    # print(np.array(kfold_dataset[1][4]).shape)
    # print(X_unlabeled)
    # print(np.array(X_unlabeled).shape)
    # print(len(kfold_dataset))
    # print(len(X_unlabeled))
#the length of Kfold_dataset is 5(the data already labelled)
#every part in kfold_dataset contains some(train、test、valid) segments, which is formed as a 
#structure  (441 × 1 × 248 × 4) (441,) (110 × 1 × 248 × 4) (110 × 5) (110,)
#totoal is 5×441 × 1 × 248 × 4

#the lenth of X_unlabeled is size 4310×
#structure is (4310 × 1 × 248 ×4 )
# #


# Encoder Network


def encoder_network(latent_dim, num_filter_ae_cls, input_labeled):
    #input_combined是做无监督,AE这一部分的input，input_labeled是做cls这一部分的
    # encoded_combined = input_combined
    encoded_labeled = input_labeled
    layers_shape = []
    #这里改了以后len(num_filter_ae_cls)只有一组需要计算的
    for i in range(len(num_filter_ae_cls)):
        #分奇偶层，奇数情况下做maxpooling
        scope_name = 'encoder_set_' + str(i + 1)
        #第一部分是编码input_combined部分的数据
        # with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, initializer=initializer):
        #     encoded_combined = tf.layers.conv2d(inputs=encoded_combined, activation=tf.nn.relu, filters=num_filter_ae_cls[i],
        #                                         name='conv_1', kernel_size=kernel_size, strides=strides,
        #                                         padding=padding)
        #第二部分的网络是编码input_labeled部分的数据
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, initializer=initializer):
            encoded_labeled = tf.layers.conv2d(inputs=encoded_labeled, activation=tf.nn.relu, filters=num_filter_ae_cls[i],
                                               name='conv_1', kernel_size=kernel_size, strides=strides, padding=padding)
        #奇数情况下做maxpooling
        if i % 2 != 0:
            # encoded_combined = tf.layers.max_pooling2d(encoded_combined, pool_size=pool_size,
            #                                               strides=pool_size, name='pool')
            encoded_labeled = tf.layers.max_pooling2d(encoded_labeled, pool_size=pool_size,
                                                          strides=pool_size, name='pool')
            # print(encoded_combined)
            # print("-----------------")
            # print("-----------------")
            # print("-----------------")
            # print("-----------------")
            # print(encoded_labeled)
            # print("-----------------")
            # print("-----------------")
            # print("-----------------")
            # print(encoded_combined.get_shape().as_list())
        #(encoderd_combined.get_shape().as_list()=[None,1,248,32])
        # layers_shape.append(encoded_combined.get_shape().as_list())
        # print(layers_shape)
        #[[None, 1, 248, 32], [None, 1, 124, 32], [None, 1, 124, 64], [None, 1, 62, 64]
        #[None, 1, 62, 128], [None, 1, 31, 128]]
        # print(i)
    # print(layers_shape)
    # print(encoderd_combined.get_shape().as_list())
    # layers_shape.append(encoded_combined.get_shape().as_list())
    # latent_combined = encoded_combined
    #latent_combined为("pool_4/MaxPool:0", shape=(?,1,31,128))
    #latent_labeled为("pool_5/MaxPool:0",shape(?,1,31,128))
    # print("latent_combined is as below:")
    # print(latent_combined)
    latent_labeled = encoded_labeled
    print("latent_labeled is as below:")
    print(latent_labeled)
    print("-----------------------")
    print("------------------------")
    print(layers_shape)
    return latent_labeled, layers_shape

# # Decoder Network


def decoder_network(latent, input_size, kernel_size, padding, activation,num_filter_ae_cls):
    decoded_combined = latent
    #num_filter_ae_cls ae_classifier的通道(filter数量即通道数量)
    num_filter_ = num_filter_ae_cls[::-1]
    #[32,32,64,64,128,128]
    print(num_filter_ae_cls[::-1])
    #[::-1] 倒序 [128, 128, 64, 64, 32, 32]

    if len(num_filter_) % 2 == 0:
        num_filter_ = sorted(set(num_filter_), reverse=True)
        for i in range(len(num_filter_)):
            decoded_combined = tf.keras.layers.UpSampling2D(name='UpSample', size=pool_size)(decoded_combined)
            scope_name = 'decoder_set_' + str(2*i)
            with tf.variable_scope(scope_name, initializer=initializer):
                decoded_combined = tf.layers.conv2d_transpose(inputs=decoded_combined, activation=activation,
                                                              filters=num_filter_[i], name='deconv_1',
                                                              kernel_size=kernel_size,
                                                              strides=strides, padding=padding)
            scope_name = 'decoder_set_' + str(2*i + 1)
            with tf.variable_scope(scope_name, initializer=initializer):
                filter_size, activation = (input_size[-1], tf.nn.sigmoid) if i == len(num_filter_) - 1 else (int(num_filter_[i] / 2), tf.nn.relu)
                if i == len(num_filter_): # change it len(num_filter_)-1 if spatial size is not dividable by 2
                    kernel_size = (1, input_size[1] - (decoded_combined.get_shape().as_list()[2] - 1) * strides)
                    padding = 'valid'
                decoded_combined = tf.layers.conv2d_transpose(inputs=decoded_combined, activation=activation,
                                                              filters=filter_size, name='deconv_1',
                                                              kernel_size=kernel_size,
                                                              strides=strides, padding=padding)
    else:
        num_filter_ = sorted(set(num_filter_), reverse=True)
        for i in range(len(num_filter_)):
            scope_name = 'decoder_set_' + str(2 * i)
            with tf.variable_scope(scope_name, initializer=initializer):
                decoded_combined = tf.layers.conv2d_transpose(inputs=decoded_combined, activation=activation,
                                                              filters=num_filter_[i], name='deconv_1',
                                                              kernel_size=kernel_size,
                                                              strides=strides, padding=padding)
            scope_name = 'decoder_set_' + str(2 * i + 1)
            with tf.variable_scope(scope_name, initializer=initializer):
                filter_size, activation = (input_size[-1], tf.nn.sigmoid) if i == len(num_filter_) - 1 else (int(num_filter_[i] / 2), tf.nn.relu)
                if i == len(num_filter_): # change it len(num_filter_)-1 if spatial size is not dividable by 2
                    kernel_size = (1, input_size[1] - (decoded_combined.get_shape().as_list()[2] - 1) * strides)
                    padding = 'valid'
                decoded_combined = tf.layers.conv2d_transpose(inputs=decoded_combined, activation=activation,
                                                              filters=filter_size, name='deconv_1',
                                                              kernel_size=kernel_size,
                                                              strides=strides, padding=padding)
                if i != len(num_filter_) - 1:
                    decoded_combined = tf.keras.layers.UpSampling2D(name='UpSample', size=pool_size)(decoded_combined)

    return decoded_combined


def classifier_mlp(latent_labeled, num_class, num_filter_cls, num_dense):
    #clsfier_mlp 为 latent_labeled的网络
    conv_layer = latent_labeled
    for i in range(len(num_filter_cls)):
        scope_name = 'cls_conv_set_' + str(i + 1)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, initializer=initializer):
            conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=num_filter_cls[i],
                                          kernel_size=kernel_size, strides=strides, padding=padding,
                                          kernel_initializer=initializer)
        if len(num_filter_cls) % 2 == 0:
            if i % 2 != 0:
                conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size,strides=pool_size, name='pool')
        else:
            if i % 2 == 0:
                conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size,strides=pool_size, name='pool')
    print("conv_layer")
    # print(conv_layer)
    #Tensor("pool_5/MaxPool:0", shape=(?, 1, 31, 128))
    #flatten 在保留axis(axis=0)的同时平移输入张量
    
    dense = tf.layers.flatten(conv_layer)

    #print(dense)
    #Tensor("flatten/Reshape:0", shape=(?, 3968))

    units = int(dense.get_shape().as_list()[-1] / 4)
    # print(units)
    # 992 *dense的shape除4
    for i in range(num_dense):
        scope_name = 'cls_dense_set_' + str(i + 1)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, initializer=initializer):
            dense = tf.layers.dense(dense, units, activation=tf.nn.relu, kernel_initializer=initializer)
        units /= 2
        print(i)
        #dense这里是0，不知道有啥用，应该是用来挑mlp参数时候用的，最后得到是0
    # sys.exit(0)
    dense_last = dense
    # print("dense before dropout")
    # print(dense)
    #Tensor("flatten/Reshape:0", shape=(?, 3968), dtype=float32)
    dense = tf.layers.dropout(dense, 0.5)
    # print("dense after dropout")
    # print(dense)
    #Tensor("dropout/Identity:0", shape=(?, 3968), dtype=float32)

    scope_name = 'cls_last_dense_'
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, initializer=initializer):
        classifier_output = tf.layers.dense(dense, num_class, name='FC_4', kernel_initializer=initializer)
        # print(classifier_output)
        # Tensor("cls_last_dense_/FC_4/BiasAdd:0", shape=(?, 5), dtype=float32)
        #output就是分类出来的5个class
    # sys.exit(0)
    return classifier_output, dense_last

def unsupervised(input_labeled, num_class , latent_dim, num_filter_ae_cls , num_dense , input_size):
    latent , layers_shape = encoder_network(latent_dim = latent_dim, num_filter_ae_cls = num_filter_ae_cls,
                                            input_labeled = input_labeled)
    decoded_output = decoder_network(latent = latent, input_size = input_size, kernel_size= kernel_size,
                                        num_filter_ae_cls=num_filter_ae_cls, activation=activation, padding=padding)

    # classifier_output, dense = classifier_mlp(latent = latent, num_class , num_filter_cls = num_filter_cls, num_dense = num_dense)
    loss_AE_label =tf.reduce_mean(tf.square(input_labeled - decoded_output))
    train_op_ae_label = tf.train.AdamOptimizer().minimize(loss_AE_label)
    return loss_AE_label, latent, train_op_ae_label
    

# def PCA_clustering():






def semi_supervised(input_labeled, input_combined, true_label, alpha, beta, num_class, latent_dim, num_filter_ae_cls, num_filter_cls, num_dense, input_size):
    #先进行encoder网络进行编码
    latent_combined, latent_labeled, layers_shape = encoder_network(latent_dim=latent_dim, num_filter_ae_cls=num_filter_ae_cls,
                                                                    input_combined=input_combined, input_labeled=input_labeled)
    #得到通过神经网络的Latent_combined和latent_labeled以及append出来的layers_shape
    decoded_output = decoder_network(latent_combined = latent_combined, input_size=input_size, kernel_size=kernel_size, activation=activation,
                                         padding=padding)
    #得到decodeNet的输出
    classifier_output, dense = classifier_mlp(latent_labeled, num_class, num_filter_cls=num_filter_cls, num_dense=num_dense)
    #classifier_output = classifier_cnn(latent_labeled, num_filter=num_filter)
    #通过mlp感知层对latent_labeled层进行一次分类
    loss_ae = tf.reduce_mean(tf.square(input_combined - decoded_output), name='loss_ae') * 100
    #ae部分的loss_function
    loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output),
                              name='loss_cls')
    #classifier部分的loss_function
    total_loss = alpha*loss_ae + beta*loss_cls
    #通过调整\alpha和\beta的参数来调整loss function的计算方法
    #total_loss = beta * loss_ae + alpha * loss_cls
    loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'EasyNet'))
    #tf.get_collection(key , scope=None)
        #   用来获取一个名称是‘key’的集合中的所有元素，返回的是一个列表，列表的顺序是按照变量放入集合中的先后;
        #   scope参数可选，表示的是名称空间（名称域），如果指定，就返回名称域中所有放入‘key’的变量的列表，不指
        #   定则返回所有变量。
    train_op_ae = tf.train.AdamOptimizer().minimize(loss_ae)
    #ae optimize 训练的operator
    train_op_cls = tf.train.AdamOptimizer().minimize(loss_cls)
    #classifier optimize 训练的oprtator
    train_op = tf.train.AdamOptimizer().minimize(total_loss)
    # train_op = train_op = tf.layers.optimize_loss(total_loss, optimizer='Adam')
    correct_prediction = tf.equal(tf.argmax(true_label, 1), tf.argmax(classifier_output, 1))
    #计算准确数
    accuracy_cls = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     #reduce_mean(
#     input_tensor,
#     axis=None,
#     keep_dims=False,
#     name=None,
#     reduction_indices=None
# )
#   tf.reduce_mean 计算张量的各个维度上的元素的平均值。
    return loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, train_op, total_loss


def get_combined_index(train_x_comb):
    x_combined_index = np.arange(len(train_x_comb))
    # print(len(x_combined_index))
    # sys.exit(0)
    np.random.shuffle(x_combined_index)
    #将combined_index做shuffle 随机index
    # print(x_combined_index)
    # sys.exit(0)
    return x_combined_index


def get_labeled_index(train_x_comb, train_x):
    labeled_index = []
    for i in range(len(train_x_comb) // len(train_x)):
        l = np.arange(len(train_x))
        np.random.shuffle(l)
        labeled_index.append(l)
    labeled_index.append(np.arange(len(train_x_comb) % len(train_x)))
    # 这里对labeled_data做了一点数据增强，使得后面可以重复训练
    return np.concatenate(labeled_index)


def ensemble_train_set(Train_X, Train_Y):
    index = np.random.choice(len(Train_X), size=len(Train_X), replace=True, p=None)
    return Train_X[index], Train_Y[index]


def loss_acc_evaluation(Test_X, Test_Y, loss_AE_label, input_labeled, k, sess):
    metrics = []
    i = 0
    print(Test_X)
    batch_size_val = 10
    # print("lenth of Test_X")
    # print(len(Test_X))
    # print(len(Test_X) // batch_size_val)
    # print(batch_size_val)
#     global i
#     global Test_X_batch
#     global Test_Y_
    for i in range(len(Test_X) // batch_size_val):
        Test_X_batch = Test_X[i * batch_size_val:(i + 1) * batch_size_val]
        Test_Y_batch = Test_Y[i * batch_size_val:(i + 1) * batch_size_val]
        loss_AE_label_ = sess.run([loss_AE_label],
                                            feed_dict={input_labeled: Test_X_batch})
        #验证集的loss和accuracy
        metrics.append([loss_AE_label_])
        print(metrics)
#     global i
    Test_X_batch = Test_X[(i + 1) * batch_size_val:]
    Test_Y_batch = Test_Y[(i + 1) * batch_size_val:]
    if len(Test_X_batch) >= 1:
        loss_AE_label_ = sess.run([loss_AE_label],
                                            feed_dict={input_labeled: Test_X_batch})
        #验证集的loss和accuracy
        metrics.append([loss_AE_label_])
    print(metrics)
    # sys.exit(0)
    mean_ = np.mean(np.array(metrics), axis=0)
    print("___________________________________")
    print(mean_)
    #print('Epoch Num {}, Loss_cls_Val {}, Accuracy_Val {}'.format(k, mean_[0], mean_[1]))
    return mean_[0]
    #把三次的loss和accuracy做一个平均传回去。

def encode_AE_data(Test_X, latent, input_labeled, sess):
    encode_result = []
    batch_s = len(Test_X)
    for i in range(len(Test_X) // batch_s):
        Test_X_batch = Test_X[i * batch_s:(i + 1) * batch_s]
        # print(np.array(Test_X_batch).shape)
        encode_result.append(sess.run(tf.nn.softmax(latent), feed_dict={input_labeled: Test_X_batch}))
        # print(np.array(encode_result).shape)
    Test_X_batch = Test_X[(i + 1) * batch_s:]
    encode_result.append(sess.run(tf.nn.softmax(latent), feed_dict={input_labeled: Test_X_batch}))
    encode_result = np.vstack(tuple(encode_result))
    return encode_result



def prediction_prob(Test_X, classifier_output, input_labeled, sess):
    prediction = []
    for i in range(len(Test_X) // batch_size):
        Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
        prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_labeled: Test_X_batch}))
    Test_X_batch = Test_X[(i + 1) * batch_size:]
    prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_labeled: Test_X_batch}))
    prediction = np.vstack(tuple(prediction))
    return prediction


#lenth of Test_X
#22
# Traceback (most recent call last):
#   File "2-Conv-Semi-AE+Cls.py", line 446, in <module>
#     label_proportions=[0.15, 0.35], num_filter=[32, 32, 64, 64])
#   File "2-Conv-Semi-AE+Cls.py", line 427, in training_all_folds
#     test_accuracy, f1_macro, f1_weight = training(kfold_dataset[i], X_unlabeled=X_unlabeled, seed=7, prop=prop, num_filter_ae_cls_all=num_filter)
#   File "2-Conv-Semi-AE+Cls.py", line 355, in training
#     loss_val, acc_val = loss_acc_evaluation(Val_X, Val_Y, loss_cls, accuracy_cls, input_labeled, true_label, k, sess)
#   File "2-Conv-Semi-AE+Cls.py", line 237, in loss_acc_evaluation
#     return mean_[0], mean_[1]
# IndexError: invalid index to scalar variable.


def train_val_split(Train_X, Train_Y_ori):
    val_index = []
    for i in range(num_class):
        # print(np.where(Train_Y_ori==i))
        # print(np.where(Train_Y_ori==i)[0])
        #This match the data to the label
        label_index = np.where(Train_Y_ori == i)[0]
        print(len(label_index))
        
        #round()方法返回浮点数x的四舍五入值。

        # print("___________")
        # print("label_index")
        # print(label_index)
        # print(label_index)
        # print(label_index[:round(0.1*len(label_index))])
        #取前10%
        val_index.append(label_index[:round(0.1*len(label_index))])
    print(val_index)
    val_index = np.hstack(tuple([label for label in val_index]))
    print(val_index)
    Val_X = Train_X[val_index]
    Val_Y_ori = Train_Y_ori[val_index]
    print(np.array(Val_Y_ori).shape)
    Val_Y = keras.utils.to_categorical(Val_Y_ori, num_classes=num_class)
    #把验证集的one-hot矩阵拼出来
    print(np.array(Val_Y).shape)
    train_index_ = np.delete(np.arange(0, len(Train_Y_ori)), val_index)
    #在训练集中去掉验证集
    print(train_index_)
    print(np.array(train_index_).shape)
    Train_X = Train_X[train_index_]
    print(np.array(Train_X).shape)
    Train_Y_ori = Train_Y_ori[train_index_]
    Train_Y = keras.utils.to_categorical(Train_Y_ori, num_classes=num_class)
    return Train_X, Train_Y, Train_Y_ori, Val_X, Val_Y, Val_Y_ori


def training(one_fold, X_unlabeled, seed, prop, num_filter_ae_cls_all, epochs_ae_cls=30):
    #each time transfer a dataset_fold to here with All unlabeled data
    Train_X = one_fold[0]
    Train_Y_ori = one_fold[1]
    # ori means its classification
    random.seed(seed)
    np.random.seed(seed)
    random_sample = np.random.choice(len(Train_X), size=round(len(Train_X)), replace=False, p=None)
    print('random_sample')
    print(random_sample)
    print(Train_X)
    Train_X1 = Train_X[random_sample]

    # 只取labeled_data中的一半做训练 通过改变size可以改变这个训练集的比例
    Train_Y_ori = Train_Y_ori[random_sample]
    #now it's only 220x
    #将验证集从训练集中抽出来
    print("before {}".format(np.shape(Train_X)))
    Train_X, Train_Y, Train_Y_ori, Val_X, Val_Y, Val_Y_ori = train_val_split(Train_X, Train_Y_ori)

    # 这里通过train_Y_ori的size改变了train_X的size

    print("after: {} and Val_X: {}".format(np.shape(Train_X), np.shape(Val_X)))
    #将验证集从训练集中单独抽出来
    Test_X = one_fold[2]
    # print(len(Test_X))
    # 110
    Test_Y = one_fold[3]
    # print(len(Test_Y))
    # 110
    Test_Y_ori = one_fold[4]
    # print(len(Test_Y_ori))
    # print(Test_Y_ori)
    # sys.exit(0)

    # 因为我做了data_augment,unlabel的tuple有30406个,label的有4285个，使用kfold的方法进行交叉训练
    # 这里我人为改少了参与训练数据的比例，其中label data 300个 unlabel data 29700个(其实根本就是混淆的)
    # 因为训练的时候loss_function完全与label无关(无mlp,只做ae),所以讲道理的话应该是与是否label无关的
    # 出于实验设计暂时还是先这样拼接 混着一部分label data进行训练吧
    random_sample_unlabel = np.random.choice(len(X_unlabeled), size=round(len(X_unlabeled)), replace=False, p=None)
    random_sample = np.random.choice(len(X_unlabeled), size=round(prop * len(X_unlabeled)), replace=False, p=None)

    # 通过改变prop来改变使用Unlabeled data的比例

    # print(len(X_unlabeled))
    # 4310
    X_unlabeled = X_unlabeled[random_sample_unlabel]
    # print(len(X_unlabeled))
    # 646
    # sys.exit(0)
    #随机选择指定量的无标签数据
    Train_X_Comb = X_unlabeled

    input_size = list(np.shape(Test_X)[1:])
    #input_size是第一个维度之后的维度
    #np.shape() 和np.array().shape的功能差不多
    # Various sets of number of filters for ensemble. If choose one set, no ensemble is implemented.
    num_filter_ae_cls_all = [[16, 16, 64, 64, 256, 256]]

# herehere here here

    unsupervised_encoded = []
    test_encoded = []

    print("X_Comb : {}".format(np.shape(Train_X_Comb)))
    Train_X = np.concatenate((Train_X,Train_X_Comb),axis=0)
    Train_X = Train_X_Comb
    # 拼成30000个点(30000 × 1 × 248 × 4)
    # 其中300个label data , 29700个unlabel data
    # print(np.shape(Train_X))
    # print(np.shape(Test_X))

    # This for loop is only for implementing ensemble
    for z in range(len(num_filter_ae_cls_all)):
        # Change the following seed to None only for Ensemble.
        tf.reset_default_graph()  # Used for ensemble
        with tf.Session() as sess:
            input_labeled = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_labeled')

            num_filter_ae_cls = num_filter_ae_cls_all[z]
            #此处配置semi_supervised内容，可以新增unsupervised内容来进行无监督启发式训练。
            # loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, train_op, total_loss = semi_supervised(
            #     input_labeled=input_labeled, input_combined=input_combined, true_label=true_label, alpha=alpha,
            #     beta=beta, num_class=num_class, latent_dim=latent_dim, num_filter_ae_cls=num_filter_ae_cls,
            #     num_filter_cls=num_filter_cls, num_dense=num_dense, input_size=input_size)
            # 配置AE模型
            loss_AE_label, latent , train_op_ae_label  = unsupervised(
                input_labeled = input_labeled, num_class = num_class, latent_dim = latent_dim, num_filter_ae_cls = num_filter_ae_cls,
                num_dense = num_dense, input_size = input_size
            )
            
            sess.run(tf.global_variables_initializer())
            #初始化
            saver = tf.train.Saver(max_to_keep=80)
            #模型保存
            # Train_X, Train_Y = ensemble_train_set(orig_Train_X, orig_Train_Y)
            val_accuracy = {-2: 0, -1: 0}
            val_loss = {-2: 10, -1: 10}
            # 对val_loss进行初始化
            num_batches = len(Train_X) // batch_size
            # alfa_val1 = [0.0, 0.0, 1.0, 1.0, 1.0]
            # beta_val1 = [1.0, 1.0, 0.1, 0.1, 0.1]
            alfa_val = 1  ## 0
            beta_val = 1
            change_to_ae = 1  # the value defines that algorithm is ready to change to joint ae-cls
            change_times = 0  # No. of times change from cls to ae-cls, which is 2 for this training strategy
            third_step = 0
            for k in range(epochs_ae_cls):
                # alfa_val = alfa_val1[k]
                # beta_val = beta_val1[k]

                #beta_val = min(((1 - 0.1) / (-epochs_ae_cls)) * k + 1, 0.1) ##
                #alfa_val = max(((1.5 - 1) / (epochs_ae_cls)) * k + 1, 1.5)

                x_combined_index = get_combined_index(train_x_comb=Train_X)
                # print(x_combined_index)
                # 646
                # print(len(x_combined_index))
                x_labeled_index = get_labeled_index(train_x_comb=Train_X_Comb, train_x=Train_X)

                # print(len(x_labeled_index))
                # print(len(x_combined_index))
                # print(num_batches)
                # sys.exit(0)

                # print(x_labeled_index)
                # 646
                # print(len(x_labeled_index))
                for i in range(num_batches):
                    # Train_X_comb=646    batch_size=100   num_batches = (Train_X_comb // batch_size = 6)
                    Comb_index_range = x_combined_index[i * batch_size: (i + 1) * batch_size]
                    # print(unlab_index_range)
                    # print(len(unlab_index_range))\
                    # print(np.array(unlab_index_range).shape)
                    # (100,)
                    # 100
                    # x_combined_index就是unlab_index_range
                    # lab_index_range = x_labeled_index[i * batch_size: (i + 1) * batch_size]
                    # label的index range 格式为(100,)
                    # print(np.array(Train_X_Comb).shape)
                    # (646,1, 248 ,4)
                    # print(Train_X_Comb)
                    # print(len(Train_X_Comb))
                    # X_ae = Train_X_Comb[unlab_index_range]
                    # (100,1, 248 , 4)
                    #抽100个unlabeled data的Index出来
                    # print(X_ae)
                    # print(len(X_ae))
                    # sys.exit(0)
                    X_cls = Train_X[Comb_index_range]
                    # 抽100个labeled data 数据(input X)的index出来
                    # Y_cls = Train_Y[lab_index_range]
                    # 100个labeled data的label
                    loss_ae_, _ = sess.run([loss_AE_label, train_op_ae_label],
                                                                     feed_dict={input_labeled: X_cls,})
                    # print('Epoch Num {}, Batches Num {}, Loss_AE {}, Loss_cls {}, Accuracy_train {}'.format
                    #       (k, i, np.round(loss_ae_, 3), np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))
                    # 训练每个batch
                # print(i)
                # 5
                # sys.exit(0)
                unlab_index_range = x_combined_index[(i + 1) * batch_size:]
                lab_index_range = x_labeled_index[(i + 1) * batch_size:]
                # X_ae = Train_X_Comb[unlab_index_range]
                X_cls = Train_X[lab_index_range]
                # Y_cls = Train_Y[lab_index_range]
                loss_ae_, _ = sess.run([loss_AE_label, train_op_ae_label],
                                                                     feed_dict={input_labeled: X_cls,})
                # print('Epoch Num {}, Batches Num {}, Loss_AE {}, Loss_cls {}, Accuracy_train {}'.format
                #       (k, i, np.round(loss_ae_, 3), np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))
                # sys.exit(0)

                print('====================================================')
                # def loss_acc_evaluation(Test_X, Test_Y, loss_AE_label, input_labeled, k, sess):
                loss_val = loss_acc_evaluation(Val_X, Val_Y, loss_AE_label, input_labeled, k, sess)
                #使用验证集来验证准确性 取val-batch里几次的平均值作为返回的loss和accuracy
                #loss_val = 1.4179071
                #acc_val = 0.7
                print(val_loss)
                #{ -2:10, -1:10}
                val_loss.update({k: loss_val})
                print(val_loss)
                print({k: loss_val})
                #{ -2:10 , -1:10, 0: 1.2811708}
                #把刚刚算得的accuracy按照 {k:{value}}的形式加到数组上去(update上去)
                print('====================================================')
                saver.save(sess, "/home/sxz/data/geolife_Data/Conv-Semi-TF-PS/" + '2/' + str(z) + '/' + str(prop), global_step=k)
                #保存模型
                # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k) + ".ckpt"
                # checkpoint = os.path.join(os.getcwd(), save_path)
                # saver.save(sess, checkpoint)
                # if alfa_val == 1:
                # beta_val += 0.05
                # 找到Max_accuracy
            print("Ensemble {}: Val_loss ae+cls Over Epochs {}: ".format(z, val_loss))

            select_train_sampleY = np.zeros((5,))
            this_fold_trainX = one_fold[0]
            this_fold_trainY = one_fold[1]
            for sel_num in range(5):

                select_train_sample_index = np.argwhere(this_fold_trainY == sel_num )

                random_sample = np.random.choice(len(select_train_sample_index), size=round(1), replace=False, p=None)
                # 这里只是在select_train_sample_index里面选出来了一个值
                print(random_sample[0])
                random_sample = select_train_sample_index[random_sample[0]][0]
                if(sel_num == 0):
                    select_train_sampleX = [this_fold_trainX[random_sample]]
                else:
                    select_train_sampleX = np.concatenate((select_train_sampleX, [this_fold_trainX[random_sample]] ))
                select_train_sampleY[sel_num] = sel_num
            # select_train_sampleY = select_train_sampleY.reshape(len(select_train_sampleY),1 ,248 ,4)
            
            
            random_sample_Un = np.random.choice(len(X_unlabeled), size=round(500), replace=False, p=None)

            select_unlabel_sample = X_unlabeled[random_sample_Un]
            
            encode_select_trainsampleX = encode_AE_data(select_train_sampleX, latent, input_labeled, sess)
            encode_unlabelsample = encode_AE_data(select_unlabel_sample, latent, input_labeled, sess)
            print(encode_select_trainsampleX)
            print(encode_unlabelsample)
            print(np.shape(encode_select_trainsampleX))
            print(np.shape(encode_unlabelsample))

            pca = PCA(n_components = 2)
            encode_select_trainsampleX = encode_select_trainsampleX.reshape(len(encode_select_trainsampleX),len(encode_select_trainsampleX[0][0])*len(encode_select_trainsampleX[0][0][0]))
            label_pca = pca.fit_transform(encode_select_trainsampleX)
            print(label_pca)
            encode_unlabelsample = encode_unlabelsample.reshape(len(encode_unlabelsample),len(encode_unlabelsample[0][0])*len(encode_unlabelsample[0][0][0]))
            unlabel_pca = pca.fit_transform(encode_unlabelsample)
            print(unlabel_pca)

            # tryr = np.vstack(( encode ))

            # sys.exit(0)

            label_pca_ = [[label_pca[:,0][i],label_pca[:,1][i]] for i in range(len(label_pca[:,0])) ]
            unlabel_pca_ = [[unlabel_pca[:,0][i], unlabel_pca[:,1][i]] for i in range(len(unlabel_pca[:,0]))]
            print(np.shape(unlabel_pca_))
            print(np.shape(label_pca_))

            choose_M = label_pca_
            unchoose_M = unlabel_pca_
            pseudo_label = []
            unlabel_index = []
            origin_index = []
            count_ = 0
            for an in range(5):
                pseudo_label.append(an)
            print(pseudo_label)


            for nn in range(50):
                if(nn == 0):
                    spread_num = 5
                else:
                    spread_num = 10
                distance_M = np.zeros((len(choose_M), len(unchoose_M)))
                for m in range(len(choose_M)):
                    for n in range(len(unchoose_M)):
                        distance_M[m][n] = math.sqrt(math.pow(choose_M[m][0] - unchoose_M[n][0],2) +math.pow(choose_M[m][1] - unchoose_M[n][1],2))
                        # 计算距离矩阵

                random_index_select = np.zeros(spread_num, dtype =int)
                for ss in range(spread_num):
                    weight = [86,108,136,151,351]
                    choice_deter = random.randint(0,351)
                    for ias in range(5):
                        if(choice_deter<=weight[ias]):
                            append_ = ias
                            break
                    # 随机选择一个该索引已经标上的数据加入到下一轮的扩点中
                    print(choice_deter)
                    wtf = np.array(pseudo_label)
                    print(np.argwhere(wtf==append_ )[0])
                    r_choose = np.random.choice(len( np.argwhere(wtf==append_ ) ),size=round(1), replace=False,p=None)
                    print(r_choose)

                    random_index_select[ss] = np.argwhere(wtf==append_)[r_choose][0]
                print(random_index_select)
                random_index_select.astype(int)
                random_index_select = np.array(random_index_select)

                # 随机选择5/10个点进行扩展
                # print(unchoose_M)
                # print(distance_M)
                print(distance_M)
                print("before expand nodes: {} ".format(len(choose_M)))
                # 进行扩展/
                for hh in range(len(random_index_select)):
#                     print(hh)
#                     print(choose_M)
                    # print(len(choose_M))
                    # print(choose_M)
#                     print(distance_M[random_index_select[hh]])
#                     print( min(distance_M[random_index_select[hh]]) )
                    print( min(distance_M[random_index_select[hh]]) )

                    min_index = np.where((distance_M[random_index_select[hh]]==min(distance_M[random_index_select[hh]])))[0]
                    # 找到这一行最小值的索引
                    min_index = min_index[0]
                    print(min_index)
                    new = unchoose_M[ min_index ]
                    # 把这个未选中的点设为new
                    choose_M.append( unchoose_M[ min_index ] )
                    # 问题在于更新之后，新加入的点在下一次扩展的时候可能会被重复考虑，并且新加入的点于自己计算会变成0
                    # 也就是加入点之后距离表不能及时更新,如果全部更新又过于compentation expensive
                    # 还是得直接删除那一列，并且在unchoose_M里面删除该数，之后通过坐标值和unlabel_pca_对比 得到在unlabel_pca_里的索引
                    # 再通过unlabel_pca_里的索引，映射到random_sample_Un里得到原始值的索引index 再X_unlabel[index]得到原始X_unlabel
                    # 删掉加入的列
                    unchoose_M = np.delete(unchoose_M,min_index,0)
                    distance_M = np.delete(distance_M,min_index,1)
                    new_dis = np.zeros((len(unchoose_M,)))
                    # print(min_index)
                    # print(unchoose_M)
                    # print(new)
                    # print(np.shape(new))
                    for n in range(len(unchoose_M)):
                        new_dis[n] = math.sqrt(math.pow(new[0] - unchoose_M[n][0],2) +math.pow(new[1] - unchoose_M[n][1],2))
                    # print(new_dis)
                    print(np.shape(distance_M))
                    # print(np.shape(new_dis))
                    distance_M = np.vstack((distance_M, new_dis))
                        # 计算距离更新之后那部分的矩阵
                        # http://whatbeg.com/2019/06/05/gpudriverupdate.html
                    # print(unchoose_M[428])
                    # unchoose_M[428] = [100,100]
                    # print(unchoose_M[428])
                    # sys.exit(0)

                    # 将选中点加入选择矩阵
                    # unchoose_M = np.delete(unchoose_M,int(np.where(( distance_M[random_index_select[hh]]==min(distance_M[random_index_select[hh]])))),)
                    # 因为要记录索引,就不直接从矩阵里删除了
                    pseudo_label.append(pseudo_label[random_index_select[hh]])
                    for i1 in range(len(unlabel_pca_)):
                        if(unlabel_pca_[i1][0] == new[0]):
                            o_i = i1
                    # print(unlabel_pca_)

                    origin_index.append( random_sample_Un[ o_i ])
                    # 这里获得原始的index，可以用来最后拼接训练矩阵
            
            print(origin_index)
            X_unlabel_pseudo = X_unlabeled[origin_index]
            print(X_unlabel_pseudo)
#             print(np.shape(select_train_sampleX))
            print(np.shape(X_unlabel_pseudo))
            X_combined = np.concatenate((select_train_sampleX, X_unlabel_pseudo))
            print(np.shape(X_combined))
            Y_combined = pseudo_label
            print(np.shape(Y_combined))
            print(np.shape(choose_M))
            X_combined = np.array(X_combined)
            Y_combined = np.array(Y_combined)
            choose_M = np.array(choose_M)
            encode_unlabelsample = np.array(encode_unlabelsample)
            with open('/home/sxz/data/geolife_Data/pseudo_data1.pickle', 'wb') as f:
                pickle.dump([X_combined, Y_combined ,choose_M, unlabel_pca], f)
            
            sys.exit(0)     
        
            # taggggg


#             unsupervised_encoded.append(encode_AE_data(one_fold[0], latent, input_labeled, sess))
#             test_encoded.append(encode_AE_data(one_fold[2], latent, input_labeled, sess))

        # ave_class_posterior = sum(class_posterior) / len(class_posterior)
        # y_pred = np.argmax(ave_class_posterior, axis=1)
        # test_accuracy = accuracy_score(Test_Y_ori, y_pred)
        # #precision = precision_score(Test_Y_ori, y_pred, average='weighted')
        # #recall = recall_score(Test_Y_ori, y_pred, average='weighted')
        # f1_macro = f1_score(Test_Y_ori, y_pred, average='macro')
        # f1_weight = f1_score(Test_Y_ori, y_pred, average='weighted')
        # print('Semi-AE+Cls Test Accuracy of the Ensemble: ', test_accuracy)
        # print('Confusion Matrix: ', confusion_matrix(Test_Y_ori, y_pred))
        # print(unsupervised_encoded[0])
        # print(np.array(unsupervised_encoded)[0].shape)
        # print(np.shape(unsupervised_encoded))

        # PCA part
#         print(unsupervised_encoded)
#         print(np.shape(unsupervised_encoded))
#         print(np.shape(unsupervised_encoded[0]))
        # sys.exit(0)
        # PCA_codearray = unsupervised_encoded[0].reshape(len(Test_X),3968)
        # print(PCA_codearray)
        # pca_ = PCA(n_components=2)
        # pca_encodeAE = pca_.fit_transform(PCA_codearray)
        # # pca_encodeAE.tofile("encodeAE.bin")
        # print(pca_encodeAE)
        # print(pca_encodeAE.dtype)
        # # Test_Y_ori.tofile("label.bin")
        # print(len(Test_Y_ori))
        # # print(Test_Y_ori.dtype)
        # sys.exit(0)
        # print(pca_encodeAE.shape)
        # x = [i[0] for i in pca_encodeAE]
        # y = [i[1] for i in pca_encodeAE]
        # print(x)
        # print(y) 
        # # plt.figure(figsize=[12,12])
        # # plt.plot(x, y,'v')
        # # plt.show()
        # # plt.savefig('test2.png')
        # km5 = KMeans(n_clusters=5, init='random',max_iter=300,n_init=10,random_state=0)
        # encode_means = km5.fit_predict(pca_encodeAE)
        # PCA part finished

        # print(np.array(encode_means).shape)
#     for i in range(5):
#         print(one_fold[i])
#     return unsupervised_encoded[0], test_encoded[0], one_fold[1] ,one_fold[4]  
# one_fold[4]是test_y_ori(非hot)
# one_fold[3]是test_y*(hot)
# one_fold[1]是train_y_ori(非hot)
def training_all_folds(label_proportions, num_filter):
    accuracy = 0
    for i in range(1):
        test_accuracy_fold = [[] for _ in range(len(label_proportions))]
        mean_std_acc = [[] for _ in range(len(label_proportions))]
        test_metrics_fold = [[] for _ in range(len(label_proportions))]
        mean_std_metrics = [[] for _ in range(len(label_proportions))]
        for index, prop in enumerate(label_proportions):
            kfold_dataset_encode = [[] for _ in range(5)]
            for i in range(len(kfold_dataset)- 4):
                Train_X_encode, Test_X_encode ,Train_label, Test_label = training(kfold_dataset[i], X_unlabeled=X_unlabeled, seed=7, prop=prop, num_filter_ae_cls_all=num_filter)
                kfold_dataset_encode[i].append(Train_X_encode)
                kfold_dataset_encode[i].append(Train_label)
                kfold_dataset_encode[i].append(Test_X_encode)
                kfold_dataset_encode[i].append(Test_label)
            print(kfold_dataset_encode)
            print(np.shape(kfold_dataset_encode))
            for i in range(4):
                print(i)
                print(np.shape(kfold_dataset_encode[0][i]))
            with open('/home/sxz/data/geolife_Data/Encoded_data_noaug.pickle', 'wb') as f:
                pickle.dump([kfold_dataset_encode], f)
            with open('/home/sxz/data/geolife_Data/Encoded_data_noaug.pickle', 'rb') as f:
                a1  = pickle.load(f)
            print(np.shape(a1))
            print(np.shape(b1))
            sys.exit(0)
            # Train_X_encode_all.tofile("encodeAE.bin")
            # label_all.tofile("label.bin")
            # encode_ = np.fromfile("encodeAE.bin",dtype=np.float32)
            encode_ = encode_.reshape(encode_len,2)
            encode_
            label_ = np.fromfile("label.bin",dtype = np.int64)
            label_
            km5 = KMeans(n_clusters=5, init='random',max_iter=300,n_init=10,random_state=0)
            encode_means = km5.fit_predict(encode_)
            mini = np.zeros((5), dtype=np.float)
            mini_coodinate = np.zeros((5,2), dtype=np.float)
            mini = [100,100,100,100,100]
            count = 0
            for j in range(5):
                for i in range(len(encode_[encode_means==j])):
                        if(mini[j] >(np.sqrt(np.sum(np.square(encode_[encode_means==j][i] - [km5.cluster_centers_[j,0],km5.cluster_centers_[j,1]] ))))):
                            mini[j] = (np.sqrt(np.sum(np.square(encode_[encode_means==j][i] - [km5.cluster_centers_[j,0],km5.cluster_centers_[j,1]] ))))
                            mini_coodinate[j] = encode_[encode_means==j][i]
            cluster_fake = np.zeros(5,)
            for j in range(5):
                for i in range(len(encode_)):
                    if(mini_coodinate[j][0] == encode_[i][0]):
                        print(i)
                        print(label_[i])
                        cluster_fake[j] = label_[i]

            label_fake = np.zeros(encode_len,)
            for i in range(5):
                label_fake[encode_means==i] =cluster_fake[i]
            count = 0
            for i in range(len(label_)):
                if(label_fake[i] == label_[i]):
                    count = count+1
            print(count)
            print('accuracy for unsupervised model is : {}'.format(count/encode_len))
            accuracy = accuracy + count/encode_len

    print('5 times average of accuracy is: {}'.format(accuracy))        
    sys.exit(0)
# end

    accuracy_all = np.array(test_accuracy_fold[index])
    mean = np.mean(accuracy_all)
    std = np.std(accuracy_all)
    mean_std_acc[index] = [mean, std]
    metrics_all = np.array(test_metrics_fold[index])
    mean_metrics = np.mean(metrics_all, axis=0)
    std_metrics = np.std(metrics_all, axis=0)
    mean_std_metrics[index] = [mean_metrics, std_metrics]
    for index, prop in enumerate(label_proportions):
        print('All Test Accuracy For Semi-AE+Cls with Prop {} are: {}'.format(prop, test_accuracy_fold[index]))
        print('Semi-AE+Cls test accuracy for prop {}: Mean {}, std {}'.format(prop, mean_std_acc[index][0], mean_std_acc[index][1]))
        print('Semi-AE+Cls test metrics for prop {}: Mean {}, std {}'.format(prop, mean_std_metrics[index][0], mean_std_metrics[index][1]))
        print('\n')
    return test_accuracy_fold, test_metrics_fold, mean_std_acc, mean_std_metrics

import time 

current = time.clock()
unsupervised_encoded = training_all_folds(
    label_proportions=[0.15], num_filter=[32, 32, 64, 64])
print("time used:{}".format(time.clock() - current))

