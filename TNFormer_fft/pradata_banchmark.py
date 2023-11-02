import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.signal as signal

fs = 250  # 采样频率
lowcut = 6.0  # 低频截止频率
highcut = 90.0  # 高频截止频率

f0 = 50.0  # 工频频率
Q = 30.0  # 陷波滤波器的品质因数
order = 4  # 滤波器阶数

b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype='band')
b1, a1 = signal.iirnotch(f0, Q, fs)


# 设置文件路径
data_path = r'F:\Devin\Dataset\Benchmark Dataset\40个目标'  # 数据文件夹路径
for num_sub in range(1,36):
    filename = 'S' + str(num_sub) + '.mat'  # 数据文件名
    srate = 250
    # 读取数据
    data = scipy.io.loadmat(data_path + '/' + filename)

    # 访问数据
    variable_name = 'data'  # MATLAB 文件中的变量名 , [64，1500，40，6], 刺激前500毫秒，刺激开始后5.5秒
    variable_data = data[variable_name]
    selected_indices = [61, 62, 63, 53, 54, 55, 56, 57, 58, 59]

    filtered_data = signal.lfilter(b, a, variable_data, axis=1)
    variable_data = signal.lfilter(b, a, filtered_data, axis=1) # filtered data

    selected_times = np.arange(0.64*srate,1.14*srate)
    start_index = 0.64  # 起始索引
    end_index = 1.14  # 结束索引（不包含在内）
    selected_times = [i for i in range(160, 410)]
    eeg = []
    for tar_num in range(variable_data.shape[2]):
        for trail_num in range(variable_data.shape[3]):
            eeg.append(variable_data[selected_indices,:,tar_num,trail_num])

    # 打印数据
    eeg = np.array(eeg)
    data = eeg[:,:,selected_times]

    # 将数据转换为二维形状（样本数，特征数）
    n_samples, n_channels, n_times = data.shape
    data_2d = data.reshape(n_samples, -1)

    # 使用MinMaxScaler进行归一化处理
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_2d)

    # 将数据恢复为三维形状（样本数，通道数，时间点数）
    data_normalized_3d = data_normalized.reshape(n_samples, n_channels, n_times)

    # 现在你可以使用data_normalized_3d进行进一步的分析和建模

    # 生成1到40的标签列表
    labels = list(range(1, 41))
    # 将标签列表循环6次
    labels = [label for _ in range(6) for label in labels]

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

    expanded_data = np.array(expanded_data)

    expanded_labels = np.repeat(labels, len(expanded_data) // len(labels))
    expanded_labels = np.array(expanded_labels)

    # 假设你的脑电epoch数据是X，对应标签是y
    X = expanded_data
    y = expanded_labels
    # 首先将数据和标签划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 然后将训练集进一步划分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 现在你有了划分好的训练集、验证集和测试集
    np.save(r'F:\Devin\Dataset\Benchmark Dataset\40个目标\train_data' + str(num_sub) + '.npy', X_train)
    np.save(r'F:\Devin\Dataset\Benchmark Dataset\40个目标\train_labels' + str(num_sub) + '.npy', y_train)

    # 保存验证集
    np.save(r'F:\Devin\Dataset\Benchmark Dataset\40个目标\val_data' + str(num_sub) + '.npy', X_val)
    np.save(r'F:\Devin\Dataset\Benchmark Dataset\40个目标\val_labels' + str(num_sub) + '.npy', y_val)

    # 保存测试集
    np.save(r'F:\Devin\Dataset\Benchmark Dataset\40个目标\test_data' + str(num_sub) + '.npy', X_test)
    np.save(r'F:\Devin\Dataset\Benchmark Dataset\40个目标\test_labels' + str(num_sub) + '.npy', y_test)