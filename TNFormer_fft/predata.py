import random
import numpy as np
import argparse
import os
import torch
import mne
from mne.io import read_raw_cnt,read_raw_edf
from mne.filter import resample
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate


path_sub = 'E:/Devin/djl/异步论文离线实验数据/AllData'
files_list_sub = os.listdir(path_sub)
num_sub = 0
for k in files_list_sub[2:]:
    # 1. 加载CNT文件
    # 2. 提取SSVEP数据
    # 3. 设置感兴趣的通道和时间窗口
    # 4. 剪切数据
    num_sub = num_sub + 1
    data_ssvep = []
    label_ssvep = []
    files_list = os.listdir("E:/Devin/djl/异步论文离线实验数据/AllData/" + k + "/ssvep/")
    save_path = "E:/Devin/djl/异步论文离线实验数据/AllData/npy/S" + str(num_sub)

    for i in files_list:
        raw = mne.io.read_raw_cnt("E:/Devin/djl/异步论文离线实验数据/AllData/" + k + "/ssvep/" + i, preload=True)  # 替换为你的CNT文件路径
        raw.filter(7, 90)
        raw.resample(sfreq=250)
        event_id = {'61': 61, '62': 62, '63': 63, '64': 64, '65': 65, '66': 66, '67': 67, '68': 68, '69': 69, '70': 70, '71': 71, '72': 72, '73': 73, '74': 74}  # 根据实际标记定义事件ID
        events, _ = mne.events_from_annotations(raw, event_id=event_id)
        target_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']  # 根据实际情况修改标签列表
        # 根据标签列表生成标签映射字典
        label_map = {label: i for i, label in enumerate(target_labels)}
        epochs = mne.Epochs(raw
                            , events
                            , event_id
                            , tmin=-1
                            , tmax=0.14
                            , baseline=None
                            , picks=['PO5', 'PO6', 'PZ', 'PO7', 'PO8', 'PO3', 'PO4', 'POZ', 'OZ']
                            , verbose=False
                            )

        # 获取Epoch数据
        epochs_data = epochs.get_data()
        # 获取Epoch标签
        epochs_labels = epochs.events[:, -1]
        # 打印每个Epoch的数据和标签
        labels_target_trail = []
        for i in range(len(epochs_data)):
            # data = epochs_data[i]
            label = epochs_labels[i]
            print(f"Epoch {i + 1} -, Label: {label}")
            labels_target_trail.append(label)

        # 5. 可以在这里对数据进行进一步的预处理、特征提取等操作

        # 6. 数据可视化（可选）
        # epochs.plot()

        # 接下来，你可以根据需要对提取的SSVEP数据进行进一步的处理和分析，例如特征提取、分类等。

        # 获取数据
        data = epochs.get_data()  # 获取SSVEP数据

        # 将数据转换为二维形状（样本数，特征数）
        n_samples, n_channels, n_times = data.shape
        data_2d = data.reshape(n_samples, -1)

        # 使用MinMaxScaler进行归一化处理
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data_2d)

        # 将数据恢复为三维形状（样本数，通道数，时间点数）
        data_normalized_3d = data_normalized.reshape(n_samples, n_channels, n_times)

        # 现在你可以使用data_normalized_3d进行进一步的分析和建模
        # 2. 标准化处理
        scaler = StandardScaler()  # 创建StandardScaler对象
        standardized_data = scaler.fit_transform(data_normalized_3d.reshape(-1, data_normalized_3d.shape[-1]))
        standardized_data = standardized_data.reshape(data_normalized_3d.shape)
        data_normalized_3d = standardized_data

        # 扩增数据
        window_size = int(0.5 * 250)  # 窗口大小，假设采样率为sampling_rate
        stride = int(0.1 * 250)  # 步长，假设采样率为sampling_rate
        expanded_data = []

        for sample in range(data_normalized_3d.shape[0]):
            t = 0
            while t + window_size < data_normalized_3d.shape[2]:
                window_data = data_normalized_3d[sample, :, t:t + window_size]
                expanded_data.append(window_data)
                t += stride

        # expanded_data = np.array(expanded_data)
        data_ssvep.append(expanded_data)
        # 假设使用目标频率作为标签，你可以根据目标频率的列表生成标签
        labels = np.repeat(labels_target_trail, len(expanded_data) // len(labels_target_trail))
        label_ssvep.append(labels)
        # 现在expanded_data包含了按照0.5秒平移窗口扩增后的数据
    data_ssvep = np.array(data_ssvep)
    label_ssvep = np.array(label_ssvep)

    # 假设data是你的4维数据
    # data的形状为 (n_epochs, n_channels, n_time_points, n_features)

    # 将前两维度变为1维
    reshaped_data = data_ssvep.reshape(-1, data_ssvep.shape[2], data_ssvep.shape[3])
    reshaped_label = label_ssvep.reshape(-1)

    # 假设你的脑电epoch数据是X，对应标签是y
    X = reshaped_data
    y = reshaped_label
    # 首先将数据和标签划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 然后将训练集进一步划分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 现在你有了划分好的训练集、验证集和测试集
    # X_train和y_train是训练集的脑电epoch数据和对应标签
    # X_val和y_val是验证集的脑电epoch数据和对应标签
    # X_test和y_test是测试集的脑电epoch数据和对应标签
    # 保存训练集

    np.save(save_path + 'train_data.npy', X_train)
    np.save(save_path + 'train_labels.npy', y_train)

    # 保存验证集
    np.save(save_path + 'val_data.npy', X_val)
    np.save(save_path + 'val_labels.npy', y_val)

    # 保存测试集
    np.save(save_path + 'test_data.npy', X_test)
    np.save(save_path + 'test_labels.npy', y_test)
