import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
import pickle
from keras.optimizers import Adam
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
x = np.ones((1, 2, 3))
a = np.transpose(x, (1, 0, 2))

filename = '/home/sxz/data/geolife_Data/e_Cross1.pickle'
with open(filename, 'rb') as f:
    Train_X, Train_Y,Test_X, Test_Y, Test_Y_ori = pickle.load(f)
print(Train_X)
# 这是表现差的
print("aaaaaaaaaaaaaaaaaaaaaaa")
filename = '/home/sxz/data/geolife_Data/Origin_data_Cross.pickle'
# 这是表现好的
with open(filename, 'rb') as f:
    Train_X, Train_Y,Test_X, Test_Y, Test_Y_ori = pickle.load(f)
print(Train_X)
# 这是伪标签拼出来的
filename = '/home/sxz/data/geolife_Data/pseudo_data1.pickle'
with open(filename, 'rb') as f:
    Train_X, Train_Y1, _ ,_ = pickle.load(f)
print(np.shape(Train_X))
print(np.shape(Train_Y1))
Train_Y = np.zeros((len(Train_Y1),5))
for i in range(len(Train_Y1)):
    Train_Y[i][Train_Y1[i]] = 1
print(Train_Y)
# Train_X = Train_X[:5]
# Train_Y = Train_Y[:5]
times = 2
acc_all = 0
acc_w_all = 0
for i in range(times):
    tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    # sess = print(tf.Session(config=tf.ConfigProto(log_device_placement=True)))

    start_time = time.clock()
    np.random.seed(7)
    random.seed(7)


    # Training and test set for GPS segments
    
#     prop = 1
#     random.seed(7)
#     np.random.seed(7)
#     tf.set_random_seed(7)
#     Train_X_Comb = Train_X
#     index = np.arange(len(Train_X))
#     np.random.shuffle(index)
#     Train_X = Train_X[index[:round(prop*len(Train_X))]]
#     Train_Y = Train_Y[index[:round(prop*len(Train_Y))]]
#     #Train_X_Comb = np.vstack((Train_X, Train_X_Unlabel))
#     random.shuffle(Train_X_Comb)



    NoClass = 5
    Threshold = 31




    # Model and Compile
    model = Sequential()
    activ = 'relu'
    model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ, input_shape=(1, 248, 4)))
    A = model.output_shape
    print(A)
    model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ))
    A = model.output_shape
    print(A)
    model.add(MaxPooling2D(pool_size=(1, 2)))
    A = model.output_shape
    print(A)
    model.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
    A = model.output_shape
    print(A)
    model.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
    A = model.output_shape
    print(A)
    model.add(MaxPooling2D(pool_size=(1, 2)))
    A = model.output_shape
    print(A)
    model.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
    A = model.output_shape
    print(A)
    model.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
    A = model.output_shape
    print(A)
    model.add(MaxPooling2D(pool_size=(1, 2)))
    A = model.output_shape
    print(A)
    model.add(Dropout(.5))
    A = model.output_shape
    print(A)
    model.add(Flatten())
    A = model.output_shape
    print(A)
    model.add(Dense(int(A[1] * 1/4.), activation=activ))
    A = model.output_shape
    print(A)
    model.add(Dropout(.5))
    A = model.output_shape
    print(A)
    model.add(Dense(NoClass, activation='softmax'))
    A = model.output_shape
    print(A)
    acc = 0
    acc_w = 0
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    offline_history = model.fit(Train_X, Train_Y, epochs=50, batch_size=256, shuffle=False,
                                validation_data=(Test_X, Test_Y))
    hist = offline_history
    print('Val_accuracy', hist.history['val_acc'])
    print('optimal Epoch: ', np.argmax(hist.history['val_acc']))
    # Saving the test and training score for varying number of epochs.
    with open('Revised_accuracy_history_largeEpoch_NoSmoothing.pickle', 'wb') as f:
        pickle.dump([hist.epoch, hist.history['acc'], hist.history['val_acc']], f)

    A = np.argmax(hist.history['val_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[A], np.max(hist.history['val_acc'])))

    # Calculating the test accuracy, precision, recall

    y_pred = np.argmax(model.predict(Test_X, batch_size=100), axis=1)
    print(y_pred)
    print(Test_Y_ori)


    print('Test Accuracy %: ', accuracy_score(Test_Y_ori, y_pred))
    print('\n')
    print('Confusin matrix: ', confusion_matrix(Test_Y_ori, y_pred))
    print('\n')
    print(classification_report(Test_Y_ori, y_pred, digits=3))
    report = classification_report(Test_Y_ori, y_pred, digits=3)
    report = report.splitlines()
    res = []
    res.append(['']+report[0].split())
    for row in report[2:-3]:
        res.append(row.split())
    lr = report[-1].split()
    res.append([' '.join(lr[:3])]+lr[3:])
    acc_w += float(res[7][0].split(' ')[2])
    for i in range(5):
        acc += float(res[i+1][1])
    print(acc)
    acc_all += acc/5
    acc_w_all += acc_w
fin = acc_all/times
fin_w = acc_w_all/times
print(fin)
print(fin_w)