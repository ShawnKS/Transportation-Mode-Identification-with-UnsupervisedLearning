import numpy as np
import pickle
import keras

# Import the final output from Instance_creation file, which is the filtered trips for all users.

filename = '/home/sxz/data/geolife_Data/paper2_trips_motion_features_NotFixedLength_woOutliers.pickle'
with open(filename, 'rb') as f:
    trip_motion_all_user_with_label, trip_motion_all_user_wo_label = pickle.load(f)
    #trip_motion_all_user_with_label = trip_motion_all_user_with_label[:1000]
    #trip_motion_all_user_wo_label = trip_motion_all_user_wo_label[:1000]
# Apply some of data preprocessing step in the paper and prepare the final input layer for deep learning

# Settings
# AccLimit = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3}
min_threshold = 20
max_threshold = 248
min_distance = 150  # Meters
min_time = 60  # Seconds
num_class = 5
new_channel = 4
# new_channel = 4
min_percentile = 0
max_percentile = 100


def take_speed_percentile(trip, min_percentile, max_percentile):
    min_threshold = np.percentile(trip[1], min_percentile)#获取百分比为percentile处的数据
    # 这里设置的min_percentile是0，max_percentile是100，即最大和最小速度，可以通过调参改变这个阈值
    max_threshold = np.percentile(trip[1], max_percentile)
    index_min = np.where(trip[1] >= min_threshold)[0]
    index_max = np.where(trip[1] <= max_threshold)[0]
    # 上面得到的index_min和index_max是两个数组,numpy.intersect1d 求两个数组的交集
    index = np.intersect1d(index_min, index_max)
    trip = trip[:, index]
    # 对选定范围内的值进行裁剪
    return trip


def trip_to_fixed_length(trip_motion_all_user, min_threshold, max_threshold, min_distance, min_time, data_type):
    if data_type == 'labeled':
        total_input = []
        total_label = []
        for index, trip in enumerate(trip_motion_all_user):
            trip, mode = trip
            trip = take_speed_percentile(trip, min_percentile=min_percentile, max_percentile=max_percentile)
            trip_length = len(trip[0])
            # all() 条件都为True的时候返回True
            if all([trip_length >= min_threshold, trip_length < max_threshold, np.sum(trip[0, :]) >= min_distance,
                    np.sum(trip[1, :]) >= min_time]):
                trip_padded = np.pad(trip, ((0, 0), (0, max_threshold - trip_length)), 'constant')
                # 0轴前后填充0个 1轴前面填充0个，后面填充(max_threshold - trip_length)个
                # np.pad() 返回一个数组
                total_input.append(trip_padded)
                total_label.append(mode)
            elif trip_length >= max_threshold:
                    quotient = trip_length // max_threshold
                    for i in range(quotient):
                        trip_truncated = trip[:, i * max_threshold:(i + 1) * max_threshold]
                        # 如果trip的length大于所设置的max_threshold，对其进行切分
                        if all([np.sum(trip_truncated[0, :]) >= min_distance, np.sum(trip_truncated[1, :]) >= min_time]):
                            total_input.append(trip_truncated)  
                            total_label.append(mode)
                    remain_trip = trip[:, (i + 1) * max_threshold:]
                    if all([(trip_length % max_threshold) > min_threshold, np.sum(remain_trip[0, :]) >= min_distance,
                            np.sum(remain_trip[1, :]) >= min_time]):
                        trip_padded = np.pad(remain_trip, ((0, 0), (0, max_threshold - trip_length % max_threshold)),
                                             'constant')
                                            #  切到最后一个进行补0
                        total_input.append(trip_padded)
                        total_label.append(mode)
        print(np.array(total_input).shape)
        print(np.array(total_label).shape)
        return np.array(total_input), np.array(total_label)

    elif data_type == 'unlabeled':
        total_input = []
        for index, trip in enumerate(trip_motion_all_user):
            trip_length = len(trip[0])
            if all([trip_length >= min_threshold, trip_length < max_threshold, np.sum(trip[0, :]) >= min_distance,
                    np.sum(trip[1, :]) >= min_time]):
                trip_padded = np.pad(trip, ((0, 0), (0, max_threshold - trip_length)), 'constant')
                total_input.append(trip_padded)
            elif trip_length >= max_threshold:
                quotient = trip_length // max_threshold
                for i in range(quotient):
                    trip_truncated = trip[:, i * max_threshold:(i + 1) * max_threshold]
                    if all([np.sum(trip_truncated[0, :]) >= min_distance, np.sum(trip_truncated[1, :]) >= min_time]):
                        total_input.append(trip_truncated)
                remain_trip = trip[:, (i + 1) * max_threshold:]
                if all([trip_length % max_threshold > min_threshold, np.sum(remain_trip[0, :]) >= min_distance,
                        np.sum(remain_trip[1, :]) >= min_time]):
                    trip_padded = np.pad(remain_trip, ((0, 0), (0, max_threshold - trip_length % max_threshold)),
                                         'constant')
                    total_input.append(trip_padded)
        return np.array(total_input)

# Max_threshold=200: 200 is the rounded median size of all trips (i.e., GPS trajectory) after removing errors and
# outliers including: 1) max speed and acceleration, (2) trip length less than 10
X_labeled, Y_labeled_ori = trip_to_fixed_length(trip_motion_all_user_with_label, min_threshold=min_threshold,
                                                                max_threshold=max_threshold, min_distance=min_distance, min_time=min_time,
                                                                data_type='labeled')
X_unlabeled = trip_to_fixed_length(trip_motion_all_user_wo_label, min_threshold=min_threshold,
                                             max_threshold=max_threshold, min_distance=min_distance, min_time=min_time, data_type='unlabeled')


def change_to_new_channel(input):
    # 做K折交
    input1 = input[:, 0:1, :]
    input2 = input[:, 3:6, :]
    input = np.concatenate((input1, input2), axis=1)
    total_input_new = np.zeros((len(input), 1, max_threshold, new_channel))
    for i in range(len(input)):
        total_input_new[i, 0, :, 0] = input[i, 0, :]
        total_input_new[i, 0, :, 1] = input[i, 1, :]
        total_input_new[i, 0, :, 2] = input[i, 2, :]
        total_input_new[i, 0, :, 3] = input[i, 3, :]
        #total_input_new[i, 0, :, 4] = input[i, 4, :]
        #total_input_new[i, 0, :, 5] = input[i, 5, :]

    return total_input_new

X_labeled = change_to_new_channel(X_labeled)
X_unlabeled = change_to_new_channel(X_unlabeled)


def min_max_scaler(input, min, max):
    """
    Min_max scaling of each channel.
    :param input:
    :param min:
    :param max:
    :return:
    """
    current_minmax = [(np.min(input[:, :, :, i]), np.max(input[:, :, :, i])) for i in range(new_channel)]
    for index, item in enumerate(current_minmax):
        input[:, :, :, index] = (input[:, :, :, index] - item[0])/(item[1] - item[0]) * (max - min) + min
    return input, current_minmax


def k_fold_stratified(X_labeled, Y_labeled_ori, fold=5):
    kfold_index = [[] for _ in range(fold)]
    for i in range(num_class):
        label_index = np.where(Y_labeled_ori == i)[0]
        for j in range(fold):
            portion = label_index[round(j*0.2*len(label_index)):round((j+1)*0.2*len(label_index))]
            kfold_index[j].append(portion)

    kfold_dataset = [[] for _ in range(num_class)]
    all_index = np.arange(0, len(Y_labeled_ori))
    for j in range(fold):
        test_index = np.hstack(tuple([label for label in kfold_index[j]]))
        Test_X = X_labeled[test_index]
        Test_Y_ori = Y_labeled_ori[test_index]
        Test_Y = keras.utils.to_categorical(Test_Y_ori, num_classes=num_class)
        train_index = np.delete(all_index, test_index)
        Train_X = X_labeled[train_index]
        Train_Y_ori = Y_labeled_ori[train_index]
        # Scaling to [0, 1]
        Train_X, current_minmax = min_max_scaler(Train_X, 0, 1)
        for index, item in enumerate(current_minmax):
            Test_X[:, :, :, index] = (Test_X[:, :, :, index] - item[0]) / (item[1] - item[0])

        kfold_dataset[j] = [Train_X, Train_Y_ori, Test_X, Test_Y, Test_Y_ori]
    return kfold_dataset

kfold_dataset = k_fold_stratified(X_labeled, Y_labeled_ori, fold=5)

X_unlabeled, _ = min_max_scaler(X_unlabeled, 0, 1)

# Test for being stratified
a = len(np.where(kfold_dataset[4][1]==0)[0])/len(kfold_dataset[4][1])

b = len(np.where(kfold_dataset[4][4]==0)[0])/len(kfold_dataset[4][4])

with open('/home/sxz/data/geolife_Data/paper2_data_for_DL_kfold_dataset_RL_augment.pickle', 'wb') as f:
    pickle.dump([kfold_dataset, X_unlabeled], f)
