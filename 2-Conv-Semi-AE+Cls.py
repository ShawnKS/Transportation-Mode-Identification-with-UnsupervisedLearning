import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
import keras
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
with open(filename, 'rb') as f:
    kfold_dataset, X_unlabeled = pickle.load(f)
    # print(len(kfold_dataset[1][1]))
    # print(np.array(kfold_dataset[1][4]).shape)
    # print(X_unlabeled)
    # print(np.array(X_unlabeled).shape)
    # print(len(kfold_dataset))
    # print(len(X_unlabeled))
#the length of Kfold_dataset is 5(the data already labelled)
#every part in kfold_dataset contains 441 segments, which is formed as a 
#structure  (441 × 1 × 248 × 4) (441,) (110 × 1 × 248 × 4) (110 × 5) (110,)
#totoal is 5×441 × 1 × 248 × 4

#the lenth of X_unlabeled is size 4310×
#structure is (4310 × 1 × 248 ×4 )
# #


# Encoder Network


def encoder_network(latent_dim, num_filter_ae_cls, input_combined, input_labeled):
    #input_combined是做无监督,AE这一部分的input，input_labeled是做cls这一部分的
    encoded_combined = input_combined
    encoded_labeled = input_labeled
    layers_shape = []
    #这里改了以后len(num_filter_ae_cls)只有一组需要计算的
    for i in range(len(num_filter_ae_cls)):
        #分奇偶层，奇数情况下做maxpooling
        scope_name = 'encoder_set_' + str(i + 1)
        #第一部分是编码input_combined部分的数据
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, initializer=initializer):
            encoded_combined = tf.layers.conv2d(inputs=encoded_combined, activation=tf.nn.relu, filters=num_filter_ae_cls[i],
                                                name='conv_1', kernel_size=kernel_size, strides=strides,
                                                padding=padding)
        #第二部分的网络是编码input_labeled部分的数据
        with tf.variable_scope(scope_name, reuse=True, initializer=initializer):
            encoded_labeled = tf.layers.conv2d(inputs=encoded_labeled, activation=tf.nn.relu, filters=num_filter_ae_cls[i],
                                               name='conv_1', kernel_size=kernel_size, strides=strides, padding=padding)
        #奇数情况下做maxpooling
        if i % 2 != 0:
            encoded_combined = tf.layers.max_pooling2d(encoded_combined, pool_size=pool_size,
                                                          strides=pool_size, name='pool')
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
        layers_shape.append(encoded_combined.get_shape().as_list())
        # print(layers_shape)
        #[[None, 1, 248, 32], [None, 1, 124, 32], [None, 1, 124, 64], [None, 1, 62, 64]
        #[None, 1, 62, 128], [None, 1, 31, 128]]
        # print(i)
    # print(layers_shape)
    # print(encoderd_combined.get_shape().as_list())
    layers_shape.append(encoded_combined.get_shape().as_list())
    latent_combined = encoded_combined
    #latent_combined为("pool_4/MaxPool:0", shape=(?,1,31,128))
    #latent_labeled为("pool_5/MaxPool:0",shape(?,1,31,128))
    print("latent_combined is as below:")
    print(latent_combined)
    latent_labeled = encoded_labeled
    print("latent_labeled is as below:")
    print(latent_labeled)
    print("-----------------------")
    print("------------------------")
    print(layers_shape)
    return latent_combined, latent_labeled, layers_shape

# # Decoder Network


def decoder_network(latent_combined, input_size, kernel_size, padding, activation):
    decoded_combined = latent_combined
    #num_filter_ae_cls ae_classifier的通道(filter数量即通道数量)
    num_filter_ = num_filter_ae_cls[::-1]
    print(num_filter_ae_cls)
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


def semi_supervised(input_labeled, input_combined, true_label, alpha, beta, num_class, latent_dim, num_filter_ae_cls, num_filter_cls, num_dense, input_size):
    #先进行encoder网络进行编码
    latent_combined, latent_labeled, layers_shape = encoder_network(latent_dim=latent_dim, num_filter_ae_cls=num_filter_ae_cls,
                                                                    input_combined=input_combined, input_labeled=input_labeled)
    #得到通过神经网络的Latent_combined和latent_labeled以及append出来的layers_shape
    decoded_output = decoder_network(latent_combined=latent_combined, input_size=input_size, kernel_size=kernel_size, activation=activation, padding=padding)
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
    print(len(x_combined_index))
    #646
    # sys.exit(0)
    np.random.shuffle(x_combined_index)
    #将combined_index做shuffle
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
    return np.concatenate(labeled_index)


def ensemble_train_set(Train_X, Train_Y):
    index = np.random.choice(len(Train_X), size=len(Train_X), replace=True, p=None)
    return Train_X[index], Train_Y[index]


def loss_acc_evaluation(Test_X, Test_Y, loss_cls, accuracy_cls, input_labeled, true_label, k, sess):
    metrics = []
    i = 0
    print(Test_X)
    batch_size_val = 10
    print("lenth of Test_X")
    print(len(Test_X))
    print(len(Test_X) // batch_size_val)
    print(batch_size_val)
#     global i
#     global Test_X_batch
#     global Test_Y_
    for i in range(len(Test_X) // batch_size_val):
        Test_X_batch = Test_X[i * batch_size_val:(i + 1) * batch_size_val]
        Test_Y_batch = Test_Y[i * batch_size_val:(i + 1) * batch_size_val]
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls],
                                            feed_dict={input_labeled: Test_X_batch,
                                                       true_label: Test_Y_batch})
        metrics.append([loss_cls_, accuracy_cls_])
#     global i
    Test_X_batch = Test_X[(i + 1) * batch_size_val:]
    Test_Y_batch = Test_Y[(i + 1) * batch_size_val:]
    if len(Test_X_batch) >= 1:
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls],
                                        feed_dict={input_labeled: Test_X_batch,
                                                   true_label: Test_Y_batch})
        metrics.append([loss_cls_, accuracy_cls_])
    print(metrics)
    mean_ = np.mean(np.array(metrics), axis=0)
    print("___________________________________")
    print(mean_)
    #print('Epoch Num {}, Loss_cls_Val {}, Accuracy_Val {}'.format(k, mean_[0], mean_[1]))
    return mean_[0], mean_[1]


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
        # print("label_index")
        # print("label_index")
        # print("label_index")
        # print("label_index")
        # print(label_index)
        # print(label_index)
        # print(label_index)
        # print(label_index)
        # print(label_index)
        # print(label_index)
        # print(label_index)
        # print(label_index)
        # print(label_index)
        # print(label_index[:round(0.1*len(label_index))])
        #取前1%
        val_index.append(label_index[:round(0.1*len(label_index))])
    print(val_index)
    val_index = np.hstack(tuple([label for label in val_index]))
    print(val_index)
    # 将不同的array压成一个array,这里是选了百分之十作为验证集
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


def training(one_fold, X_unlabeled, seed, prop, num_filter_ae_cls_all, epochs_ae_cls=20):
    #each time transfer a dataset_fold to here with All unlabeled data
    Train_X = one_fold[0]
    Train_Y_ori = one_fold[1]
    # ori means its classification
    random.seed(seed)
    np.random.seed(seed)
    random_sample = np.random.choice(len(Train_X), size=round(0.5*len(Train_X)), replace=False, p=None)
    print('random_sample')
    print(random_sample)
    print(Train_X)
    Train_X1 = Train_X[random_sample]

    #This random_sample generate a (220,) matrix which will random make a 
    #(220,1,248,4)matrix from (441,1,248,4) if we use the statement A = A[random_sample]

    # print(np.array(Train_X1).shape)
    # print(np.array(Train_X).shape)
    # print(np.array(random_sample).shape)

    Train_Y_ori = Train_Y_ori[random_sample]
    #now it's only 220x
    #将验证集从训练集中抽出来
    Train_X, Train_Y, Train_Y_ori, Val_X, Val_Y, Val_Y_ori = train_val_split(Train_X, Train_Y_ori)
    #将验证集从训练集中单独抽出来
    Test_X = one_fold[2]
    # print(len(Test_X))
    # 110
    Test_Y = one_fold[3]
    # print(len(Test_Y))
    # 110
    Test_Y_ori = one_fold[4]
    random_sample = np.random.choice(len(X_unlabeled), size=round(prop * len(X_unlabeled)), replace=False, p=None)
    # print(len(X_unlabeled))
    # 4310
    X_unlabeled = X_unlabeled[random_sample]
    # print(len(X_unlabeled))
    # 646
    # sys.exit(0)
    #随机选择指定量的无标签数据
    Train_X_Comb = X_unlabeled
    #别忘了写计网的前端
    input_size = list(np.shape(Test_X)[1:])
    #input_size是第一个维度之后的维度
    #np.shape() 和np.array().shape的功能差不多
    # Various sets of number of filters for ensemble. If choose one set, no ensemble is implemented.
    num_filter_ae_cls_all = [[32, 32], [32, 32, 64], [32, 32, 64, 64], [32, 32, 64, 64, 128],
                             [32, 32, 64, 64, 128, 128], [32, 32, 64, 64, 128, 128], [32, 32, 64, 64, 128, 128]]
    num_filter_ae_cls_all = [[32, 32, 64, 64, 128, 128]]
    class_posterior = []

    # This for loop is only for implementing ensemble
    # 以下loop实现了ensemble(你懂得)
    for z in range(len(num_filter_ae_cls_all)):
        # Change the following seed to None only for Ensemble.
        tf.reset_default_graph()  # Used for ensemble
        with tf.Session() as sess:
            input_labeled = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_labeled')
            input_combined = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_combined')
            true_label = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label')
            alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
            beta = tf.placeholder(tf.float32, shape=(), name='beta')

            num_filter_ae_cls = num_filter_ae_cls_all[z]
            #此处配置semi_supervised内容，可以新增unsupervised内容来进行无监督启发式训练。
            loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, train_op, total_loss = semi_supervised(
                input_labeled=input_labeled, input_combined=input_combined, true_label=true_label, alpha=alpha,
                beta=beta, num_class=num_class, latent_dim=latent_dim, num_filter_ae_cls=num_filter_ae_cls,
                num_filter_cls=num_filter_cls, num_dense=num_dense, input_size=input_size)
            sess.run(tf.global_variables_initializer())
            #初始化
            saver = tf.train.Saver(max_to_keep=20)
            #模型保存
            # Train_X, Train_Y = ensemble_train_set(orig_Train_X, orig_Train_Y)
            val_accuracy = {-2: 0, -1: 0}
            val_loss = {-2: 10, -1: 10}
            num_batches = len(Train_X_Comb) // batch_size
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

                x_combined_index = get_combined_index(train_x_comb=Train_X_Comb)
                # print(x_combined_index)
                # 646
                # print(len(x_combined_index))
                x_labeled_index = get_labeled_index(train_x_comb=Train_X_Comb, train_x=Train_X)
                # print(x_labeled_index)
                # 646
                # print(len(x_labeled_index))
                for i in range(num_batches):
                    unlab_index_range = x_combined_index[i * batch_size: (i + 1) * batch_size]
                    lab_index_range = x_labeled_index[i * batch_size: (i + 1) * batch_size]
                    X_ae = Train_X_Comb[unlab_index_range]
                    X_cls = Train_X[lab_index_range]
                    Y_cls = Train_Y[lab_index_range]
                    loss_ae_, loss_cls_, accuracy_cls_, _ = sess.run([loss_ae, loss_cls, accuracy_cls, train_op],
                                                                     feed_dict={alpha: alfa_val, beta: beta_val,
                                                                                input_combined: X_ae,
                                                                                input_labeled: X_cls,
                                                                                true_label: Y_cls})
                    #print('Epoch Num {}, Batches Num {}, Loss_AE {}, Loss_cls {}, Accuracy_train {}'.format
                          #(k, i, np.round(loss_ae_, 3), np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

                unlab_index_range = x_combined_index[(i + 1) * batch_size:]
                lab_index_range = x_labeled_index[(i + 1) * batch_size:]
                X_ae = Train_X_Comb[unlab_index_range]
                X_cls = Train_X[lab_index_range]
                Y_cls = Train_Y[lab_index_range]
                loss_ae_, loss_cls_, accuracy_cls_, _ = sess.run([loss_ae, loss_cls, accuracy_cls, train_op],
                                                                 feed_dict={alpha: alfa_val, beta: beta_val,
                                                                            input_combined: X_ae,
                                                                            input_labeled: X_cls, true_label: Y_cls})
                print('Epoch Num {}, Batches Num {}, Loss_AE {}, Loss_cls {}, Accuracy_train {}'.format
                      (k, i, np.round(loss_ae_, 3), np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

                print('====================================================')
                loss_val, acc_val = loss_acc_evaluation(Val_X, Val_Y, loss_cls, accuracy_cls, input_labeled, true_label, k, sess)
                val_loss.update({k: loss_val})
                val_accuracy.update({k: acc_val})
                print('====================================================')
                saver.save(sess, "/home/sxz/data/geolife_Data/Conv-Semi-TF-PS/" + '2/' + str(z) + '/' + str(prop), global_step=k)
                # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k) + ".ckpt"
                # checkpoint = os.path.join(os.getcwd(), save_path)
                # saver.save(sess, checkpoint)
                # if alfa_val == 1:
                # beta_val += 0.05

                if all([change_to_ae, val_accuracy[k] < val_accuracy[k - 1], val_accuracy[k] < val_accuracy[k - 2]]):
                    # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k-1) + ".ckpt"
                    # checkpoint = os.path.join(os.getcwd(), save_path)
                    max_acc = max(val_accuracy.items(), key=lambda k: k[1])[0]
                    save_path = "/home/sxz/data/geolife_Data/Conv-Semi-TF-PS/" + '2/' + str(z) + '/' + str(prop) + '-' + str(max_acc)
                    saver.restore(sess, save_path)
                    alfa_val = 1.0
                    beta_val = 0.1
                    num_epoch_cls_only = k
                    change_times += 1
                    change_to_ae = 1
                    key = 'change_' + str(k)
                    val_accuracy.update({key: val_accuracy[k]})
                    val_loss.update({key: val_loss[k]})
                    #val_accuracy.update({k: val_accuracy[max_acc] - 0.001}) ##
                    #val_loss.update({k: val_loss[max_acc] + 0.001})  ##

                elif all([not change_to_ae, val_accuracy[k] < val_accuracy[k - 1],
                          val_accuracy[k] < val_accuracy[k - 2]]):
                    # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k - 1) + ".ckpt"
                    # #checkpoint = os.path.join(os.getcwd(), save_path)
                    max_acc = max(val_accuracy.items(), key=lambda k: k[1])[0]
                    # saver.restore(sess, "/Conv-Semi-TF-PS/" + str(prop) + '/' + str(max_acc) + ".ckpt")
                    save_path = "/home/sxz/data/geolife_Data/Conv-Semi-TF-PS/" + '2/' + str(z) + '/' + str(prop) + '-' + str(max_acc)
                    saver.restore(sess, save_path)
                    num_epoch_ae_cls = k - num_epoch_cls_only - 1
                    alfa_val = 1.5
                    beta_val = 0.2
                    change_times += 1  ##
                    change_to_ae = 1
                    key = 'change_' + str(k)
                    val_accuracy.update({key: val_accuracy[k]})
                    val_loss.update({key: val_loss[k]})
                    #val_accuracy.update({k: val_accuracy[max_acc] - 0.001})  ##
                    #val_loss.update({k: val_loss[max_acc] + 0.001})  ##
                if change_times == 2: ##
                    break

            print("Ensemble {}: Val_Accu ae+cls Over Epochs {}: ".format(z, val_accuracy))
            print("Ensemble {}: Val_loss ae+cls Over Epochs {}: ".format(z, val_loss))
            class_posterior.append(prediction_prob(Test_X, classifier_output, input_labeled, sess))

        ave_class_posterior = sum(class_posterior) / len(class_posterior)
        y_pred = np.argmax(ave_class_posterior, axis=1)
        test_accuracy = accuracy_score(Test_Y_ori, y_pred)
        #precision = precision_score(Test_Y_ori, y_pred, average='weighted')
        #recall = recall_score(Test_Y_ori, y_pred, average='weighted')
        f1_macro = f1_score(Test_Y_ori, y_pred, average='macro')
        f1_weight = f1_score(Test_Y_ori, y_pred, average='weighted')
        print('Semi-AE+Cls Test Accuracy of the Ensemble: ', test_accuracy)
        print('Confusion Matrix: ', confusion_matrix(Test_Y_ori, y_pred))

    return test_accuracy, f1_macro, f1_weight

def training_all_folds(label_proportions, num_filter):
    test_accuracy_fold = [[] for _ in range(len(label_proportions))]
    mean_std_acc = [[] for _ in range(len(label_proportions))]
    test_metrics_fold = [[] for _ in range(len(label_proportions))]
    mean_std_metrics = [[] for _ in range(len(label_proportions))]
    for index, prop in enumerate(label_proportions):
        for i in range(len(kfold_dataset)):
            test_accuracy, f1_macro, f1_weight = training(kfold_dataset[i], X_unlabeled=X_unlabeled, seed=7, prop=prop, num_filter_ae_cls_all=num_filter)
            test_accuracy_fold[index].append(test_accuracy)
            test_metrics_fold[index].append([f1_macro, f1_weight])
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

test_accuracy_fold, test_metrics_fold, mean_std_acc, mean_std_metrics = training_all_folds(
    label_proportions=[0.15, 0.35], num_filter=[32, 32, 64, 64])


