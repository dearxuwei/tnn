from scipy.io import loadmat
import numpy as np
import math
from scipy.signal import cheb1ord, filtfilt, cheby1
import os
end_point = 1140
import h5py

Fs = 1000. # sampling freq

################################################################################
# prepare training or test dataset
# input:
#       subj: subject index
#       runs: trial index
#       tw: time window length
#       cl: # of total frequency classes
#       permutations: channel indexes
# output:
#       x: dataset [?,tw,ch], ?=cl*runs*samples
#       y: labels [?,1]
#
def prepare_data_as(subj,runs,tw, flag, root_path, cl=2,permutation=[0,1,2,3,4,5,6,7,8]):

    step = 40 # 40ms
    ch = len(permutation) # # of channels
    x = np.array([],dtype=np.float32).reshape(0,tw,ch) # data
    y = np.zeros([0], dtype=np.int32)  # true label
    # 构建文件路径
    file_name = 'S' + str(subj[0]) + 'C.mat'
    file_path = os.path.join(root_path, file_name)
    # 用正斜杠替换反斜杠
    file_path = file_path.replace('\\', '/')
    # 加载 .mat 文件
    # file = loadmat(file_path)['data']
    with h5py.File(file_path, 'r') as data:
        file = data['data'][:]  # 读取数据集
    file = file.transpose(3,2,1,0)
    # file = loadmat(r'F:/Devin/Dataset/Benchmark Dataset/40个目标/S'+str(subj)+'.mat')['data']
    # file = loadmat('F:Devin\\Dataset\\Benchmark Dataset\\40\\S3.mat')['data']
    for run_idx in runs:
        for freq_idx in range(2):
            raw_data = file[freq_idx, permutation, 140:end_point, run_idx].T

            if flag == "TRAIN":
                n_samples = int(math.floor((raw_data.shape[0] - tw) / step))
            else:
                n_samples = 1

            _x = np.zeros([n_samples, tw, ch], dtype=np.float32)
            _y = np.ones([n_samples], dtype=np.int32) * freq_idx
            for i in range(n_samples):
                _x[i, :, :] = raw_data[i * step:i * step + tw, :]

            x = np.append(x, _x, axis=0)  # [?,tw,ch], ?=runs*cl*samples
            y = np.append(y, _y)  # [?,1]

    # x = filter(x)
    # x_ = np.zeros([x.shape[0], ch, tw])  # data
    # x_ = x.transpose(0, 2, 1)
    print('S'+str(subj)+'|x',x.shape)
    return x, y

def normalize(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            _x = x[i,:,j]
            x[i,:,j] = (_x - _x.mean())/_x.std(ddof=1)
    return x

## prepossing by Chebyshev Type I filter
from scipy.signal import butter, lfilter

def filter(x):
    nyq = 0.5 * Fs
    Wp = [6 / nyq, 90 / nyq]
    Ws = [4 / nyq, 100 / nyq]

    # Calculate the order of the Butterworth filter
    # N, Wn = butter(N=3, Wn=[Wp[0], Wp[1]], btype='band')

    # Get the coefficients for the filter
    b, a = butter(3, Wp, btype='band')

    # Apply the Butterworth filter to each channel of the input signal
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            _x = x[i, :, j]
            x[i, :, j] = lfilter(b, a, _x)

    return x

