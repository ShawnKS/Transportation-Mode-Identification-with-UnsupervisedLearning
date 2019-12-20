import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import heapq

# filename = '/home/sxz/data/geolife_Data/Origin_data_Cross.pickle'
# with open(filename, 'rb') as f:
#     Train_X1, Train_Y1,Test_X1, Test_Y1, Test_Y_ori1 = pickle.load(f)
filename = '/home/sxz/data/geolife_Data/My_data_for_DL_kfold_dataset_RL.pickle'
with open(filename, 'rb') as f:
    kfold_dataset1, unlabel = pickle.load(f)

print(np.shape(unlabel))
random_sample = np.random.choice(len(unlabel), size=int(0.1*len(unlabel)), replace=True, p=None)
unlabel = unlabel[random_sample]
# sys.exit(0)

filename = '/home/sxz/data/geolife_Data/My_data_for_DL_kfold_dataset_RL.pickle'
with open(filename, 'rb') as f:
    kfold_dataset, unlabel = pickle.load(f)

filename = '/home/sxz/data/geolife_Data/pseudo_data4.pickle'
with open(filename, 'rb') as f:
    Train_Xp, Train_Yp = pickle.load(f)
# print(kfold_dataset[0][1])
# print(len(kfold_dataset[0][1][kfold_dataset[0][1]==0]))
# print(len(kfold_dataset[0][1][kfold_dataset[0][1]==1]))
# print(len(kfold_dataset[0][1][kfold_dataset[0][1]==2]))
# print(len(kfold_dataset[0][1][kfold_dataset[0][1]==3]))
# print(len(kfold_dataset[0][1][kfold_dataset[0][1]==4]))
# sys.exit(0)
times = 1
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
        prop = 12000/14000
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
            model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ, input_shape=(1, 248, 4)))
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
        Train_X = Train_Xp

        random_sample = np.random.choice(len(Train_X), size=int(prop*len(Train_X)), replace=True, p=None)
        Train_X = Train_X[random_sample]
        ori = Train_Yp



        Train_Y = np.zeros([len(Train_Yp) , 5])
        
        for k in range(len(Train_Yp)):
            Train_Y[k][ori[k]] = 1
        Train_Y = Train_Y[random_sample]
        ori = ori[random_sample]

        # # 以下是只抽5个样本出来训练的结果
        # index = np.zeros((5,),dtype = int)
        # for i in range(5):
        #     print(i)
        #     print(np.where( ori == i )[0])
        #     index[i] = np.where( ori == i)[0][0]
        # print(index)
        # Train_X = Train_X[index]
        # Train_Y = Train_Y[index]
        
        # Train_X_tmp = Train_X
        # Train_Y_tmp = ori[index]

        # print(Train_Y)
        # print(np.where(Train_Y==[1,0,0,0,0] ))
        # print(Train_Y[k][2])
        # print(Train_Y[k][3])
        # print(Train_Y[k][4])
        # sys.exit(0)
        Test_X = kfold_dataset1[i][2]
        Test_Y = kfold_dataset1[i][3]
        Test_Y_ori = kfold_dataset1[i][4]

        print(np.shape(Train_Y))
        print(np.shape(Train_X))
        # sys.exit(0)
        
        
        y_pred_all = np.zeros((ensemble_num,len(Test_X)))

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

            r = model_all[i2].predict(unlabel,batch_size=1000)
            print(r)
            print(np.argmax(r,axis=1))
            mark = np.argmax(r,axis=1)

            _0index = np.where(mark == 0)[0]
            print(np.shape(mark))
            print(np.shape(_0index))
            print(_0index)
            print(r[_0index][:,0])
            _0array = r[_0index][:,0].tolist()
            max_num_index_0 = map(_0array.index, heapq.nlargest(20,_0array))
            temp = list(max_num_index_0)
            print(np.shape(_0index))
            print(temp)
            print(_0index[temp])
            #_0index[temp]就是置信度最高的指定个数的unlabel的点

            u0data = unlabel[_0index[temp]]

            print(np.shape(u0data))

            _1index = np.where(mark == 1)[0]
            _1array = r[_1index][:,1].tolist()
            max_num_index_1 = map(_1array.index, heapq.nlargest(20,_1array))
            temp = list(max_num_index_1)
            u1data = unlabel[_1index[temp]]
            print(np.shape(u1data))


            _2index = np.where(mark == 2)[0]
            _2array = r[_2index][:,2].tolist()
            max_num_index_2 = map(_2array.index, heapq.nlargest(20,_2array))
            temp = list(max_num_index_2)
            u2data = unlabel[_2index[temp]]
            print(np.shape(u2data))

            _3index = np.where(mark == 3)[0]
            _3array = r[_3index][:,3].tolist()
            max_num_index_3 = map(_3array.index, heapq.nlargest(20,_3array))
            temp = list(max_num_index_3)
            u3data = unlabel[_3index[temp]]
            print(np.shape(u3data))


            _4index = np.where(mark == 4)[0]
            _4array = r[_4index][:,4].tolist()
            max_num_index_4 = map(_4array.index, heapq.nlargest(20,_4array))
            temp = list(max_num_index_4)
            u4data = unlabel[_4index[temp]]
            print(np.shape(u4data))

            unlabel_t = []
            unlabel_t = np.vstack((u0data,u1data))
            unlabel_t = np.vstack((unlabel_t,u2data))
            unlabel_t = np.vstack((unlabel_t,u3data))
            unlabel_t = np.vstack((unlabel_t,u4data))
            unlabel_Y = np.zeros((100,),dtype = int)
            unlabel_Y[:20] = 0
            unlabel_Y[20:40] = 1
            unlabel_Y[40:60] = 2
            unlabel_Y[60:80] = 3
            unlabel_Y[80:100] = 4
            # unlabel_t = np.vstack((unlabel_t, Train_X_tmp))
            # print(unlabel_Y)
            # unlabel_Y = np.hstack((unlabel_Y,Train_Y_tmp))
            # print(unlabel_Y)
            # print(np.shape(unlabel_t))
            # print(np.shape(unlabel_Y))
            # print(unlabel_Y[4999])
            # sys.exit(0)
            # _1index = np.where(mark == 1)
            # _2index = np.where(mark == 2)
            # _3index = np.where(mark == 3)
            # _4index = np.where(mark == 4)
            # max_1000_0 = map(_0index.index,heapq.nlargest(1000,))
            # print(len(np.argmax(r,axis=1)[np.argmax(r,axis=1) == 0]))
            # print(len(np.argmax(r,axis=1)[np.argmax(r,axis=1) == 1]))
            # print(len(np.argmax(r,axis=1)[np.argmax(r,axis=1) == 2]))
            # print(len(np.argmax(r,axis=1)[np.argmax(r,axis=1) == 3]))
            # print(len(np.argmax(r,axis=1)[np.argmax(r,axis=1) == 4]))
            # with open('/home/sxz/data/geolife_Data/pseudo_data4.pickle', 'wb') as f:
            #     pickle.dump([unlabel_t, unlabel_Y], f)
                # pseudo_data3是真正的纯粹伪标签
            # 每一类选择置信度最高的那一个点
            # sys.exit(0)

            # Calculating the test accuracy, precision, recall
            y_pred_all[i2] = np.argmax(model_all[i2].predict(Test_X, batch_size=100), axis=1)
            print('Test Accuracy %: ', accuracy_score(Test_Y_ori, y_pred_all[i2]))
            print('\n')
            print('Confusin matrix: ', confusion_matrix(Test_Y_ori, y_pred_all[i2]))
            print('\n')
            sys.exit(0)



        # print(y_pred_all)
        y_pred_ens = np.zeros((len(Test_Y_ori),5))
        for jj in range(ensemble_num):
            for jji in range(len(Test_X)):
                y_pred_ens[jji][int(y_pred_all[jj][jji])] += 1
        # print(y_pred_ens)
        y_pred_final = np.argmax(y_pred_ens, axis = 1)
        print(Test_Y_ori)

        print('Test Accuracy %: ', accuracy_score(Test_Y_ori, y_pred_final))
        print('\n')
        print('Confusin matrix: ', confusion_matrix(Test_Y_ori, y_pred_final))
        print('\n')
        sys.exit(0)
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