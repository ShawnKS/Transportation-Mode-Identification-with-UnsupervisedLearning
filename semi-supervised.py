import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# filename = '/home/sxz/data/geolife_Data/Origin_data_Cross.pickle'
# with open(filename, 'rb') as f:
#     Train_X1, Train_Y1,Test_X1, Test_Y1, Test_Y_ori1 = pickle.load(f)
filename = '/home/sxz/data/geolife_Data/kfold_dataset_encode.pickle'
with open(filename, 'rb') as f:
    kfold_dataset = pickle.load(f)

times = 5
acc_all = 0
acc_w_all = 0
for T in range(times):

    for i in range(len(kfold_dataset)):
        tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        # sess = print(tf.Session(config=tf.ConfigProto(log_device_placement=True)))
        start_time = time.clock()
        np.random.seed(7)
        random.seed(7)


        # Training and test set for GPS segments
        prop = 1
        random.seed(7)
        np.random.seed(7)
        tf.set_random_seed(7)
        # index = np.arange(len(Train_X))
        # np.random.shuffle(index)
        # Train_X = Train_X[index[:round(prop*len(Train_X))]]
        # Train_Y = Train_Y[index[:round(prop*len(Train_Y))]]
        # #Train_X_Comb = np.vstack((Train_X, Train_X_Unlabel))
        # random.shuffle(Train_X_Comb)

        ensemble_num = 1
        NoClass = 5
        Threshold = 31



        model_all = []
        for i1 in range(ensemble_num):
        # Model and Compile
            model = Sequential()
            activ = 'relu'
            model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ, input_shape=(1, 31, 128)))
            A = model.output_shape
            # print(A)
            model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ))
            A = model.output_shape
            # print(A)
            model.add(MaxPooling2D(pool_size=(1, 2)))
            A = model.output_shape
            # print(A)
            model.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
            A = model.output_shape
            # print(A)
            model.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
            A = model.output_shape
            # print(A)
            model.add(MaxPooling2D(pool_size=(1, 2)))
            A = model.output_shape
            # print(A)
            model.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
            A = model.output_shape
            # print(A)
            model.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
            A = model.output_shape
            # print(A)
            model.add(MaxPooling2D(pool_size=(1, 2)))
            A = model.output_shape
            # print(A)
            model.add(Dropout(.5))
            A = model.output_shape
            # print(A)
            model.add(Flatten())
            A = model.output_shape
            model.add(Dense(int(A[1] * 1/4.), activation=activ))
            A = model.output_shape
            model.add(Dropout(.5))
            A = model.output_shape
            model.add(Dense(NoClass, activation='softmax'))
            A = model.output_shape
            model_all.append(model)
        print(model_all)
        acc = 0
        acc_w =0
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        Train_X = kfold_dataset[i][0]
        ori = kfold_dataset[i][1]
        Train_Y = np.zeros([len(kfold_dataset[i][0]) , 5])
        
        for k in range(len(kfold_dataset[i][0])):
            Train_Y[k][ori[k]] = 1
        print(ori)
        print(Train_Y)
        sys.exit(0)
        # print(Train_Y)
        # print(np.where(Train_Y==[1,0,0,0,0] ))
        # print(Train_Y[k][2])
        # print(Train_Y[k][3])
        # print(Train_Y[k][4])
        # sys.exit(0)
        Test_X = kfold_dataset[i][2]
        Test_Y = kfold_dataset[i][3]
        Test_Y_ori = kfold_dataset[i][4]
        y_pred_all = np.zeros((ensemble_num,len(Test_X)))

        print(Train_X)
        print("...........")
        print(Train_Y)
        print("...........")
        print(Test_X)
        print("...........")
        print(Test_Y)
        print("...........")
        print(Test_Y_ori)
        print("...........")

        for i2 in range(ensemble_num):
            model_all[i2].compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            offline_history = model_all[i2].fit(Train_X, Train_Y, epochs=50, batch_size=512, shuffle=False,
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
            y_pred_all[i2] = np.argmax(model_all[i2].predict(Test_X, batch_size=100), axis=1)
        
        # print(y_pred_all)
        y_pred_ens = np.zeros((len(Test_Y_ori),5))
        for jj in range(ensemble_num):
            for jji in range(len(Test_X)):
                y_pred_ens[jji][int(y_pred_all[jj][jji])] += 1
        # print(y_pred_ens)
        y_pred_final = np.argmax(y_pred_ens, axis = 1)
        print(y_pred_final)
        print(len(y_pred_final))
        print(Test_Y_ori)
        print(len(Test_Y_ori))

        print('Test Accuracy %: ', accuracy_score(Test_Y_ori, y_pred_final))
        print('\n')
        print('Confusin matrix: ', confusion_matrix(Test_Y_ori, y_pred_final))
        print('\n')
        print(classification_report(Test_Y_ori, y_pred_final, digits=3))
        report = classification_report(Test_Y_ori, y_pred_final, digits=3)
        report = report.splitlines()
        res = []
        res.append(['']+report[0].split())
        for row in report[2:-3]:
            res.append(row.split())
        lr = report[-1].split()
        res.append([' '.join(lr[:3])]+lr[3:])
        acc_w += float(res[7][0].split(' ')[2])
        acc_w_all += acc_w
        for ii in range(5):
            acc += float(res[ii+1][1])
        print(acc)
        acc_all += acc/5
fin = acc_all/(times*5)
fin_w = acc_w_all/(times*5)
print(fin)
print(fin_w)