import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.fft
# from layers.Embed import DataEmbedding
# from layers.Conv_Blocks import Inception_Block_V1
from scipy.io import loadmat
import numpy as np
import time
import math
from scipy.signal import cheb1ord, filtfilt, cheby1
end_point = 1375
Fs = 250. # sampling freq
## for debug
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    print('top_list:',top_list)
    return period, abs(xf).mean(-1)[:, top_list]



def prepare_data_as(subj=1,runs=[2,3,4,5],tw=375+160, cl=40,permutation=[47,53,54,55,56,57,60,61,62]):

    all_freqs = loadmat('F:\\DeepLearning\\paper\\bi-SiamCA-main\\data\\Freq_Phase.mat')['freqs'][0] # true freq
    step = (end_point-160-tw) # 40ms, origin:10
    ch = len(permutation) # # of channels
    x = np.array([],dtype=np.float32).reshape(0,tw,ch) # data
    y = np.zeros([0],dtype=np.int32) # true label
    # file = loadmat('./data/S'+str(subj)+'.mat')['data']
    file = loadmat('F:\\DeepLearning\\paper\\bi-SiamCA-main\\data\\S1.mat')['data']
    # load 佳乐姐的数据

    for run_idx in runs:
        for freq_idx in range(cl):
            raw_data = file[permutation,160:end_point,freq_idx,run_idx].T
            n_samples = int(math.floor((raw_data.shape[0]-tw)/step))
            _x = np.zeros([n_samples,tw,ch],dtype=np.float32)
            _y = np.ones([n_samples],dtype=np.int32) * freq_idx
            for i in range(n_samples):
                _x[i,:,:] = raw_data[i*step:i*step+tw,:]

            x = np.append(x,_x,axis=0) # [?,tw,ch], ?=runs*cl*samples
            y = np.append(y,_y)        # [?,1]

    x = filter(x)
    print('S'+str(subj)+'|x',x.shape)

    xt = torch.from_numpy(x)
    B, T, N = xt.size()
    period_list, period_weight = FFT_for_Period(xt, 14)

    period_list = torch.arange(8,15,0.5)
    period_weight = torch.ones(14)
    #period_list
    # for timenet, period_list =
    # for i in range(14):
    #     i = 0
    #     period = period_list[i]
    #     if (T) % period != 0:
    #         length = (((T) // period) + 1) * period
    #         length_int = length.int()
    #         a = x.shape[0]
    #         padding = torch.zeros([x.shape[0], (length_int - (T)), x.shape[2]]).to(xt.device)
    #         out = torch.cat([xt, padding], dim=1)
    #     else:
    #         length = (T)
    #         out = x
    #     out = out.reshape(B, (length // period).int(), period.int(),
    #                       N).permute(0, 3, 1, 2).contiguous()
    #     # 2D conv: from 1d Variation to 2d Variation
    #
    #
    #     self.conv = nn.Sequential(
    #         Inception_Block_V1(configs.d_model, configs.d_ff,
    #                            num_kernels=configs.num_kernels),
    #         nn.GELU(),
    #         Inception_Block_V1(configs.d_ff, configs.d_model,
    #                            num_kernels=configs.num_kernels)
    #     )
    #

    return x, y, all_freqs

# def time_shape(x,period_list,period_weight)
#     y = x+1
#
#
#     return y
    # # forward step debug here
    # res = []
    # for i in range(14):
    #     period = period_list[i]
    #     # padding


## prepossing by Chebyshev Type I filter


def filter(x):
    nyq = 0.5 * Fs
    Wp = [6/nyq, 90/nyq]
    Ws = [4/nyq, 100/nyq]
    N, Wn=cheb1ord(Wp, Ws, 3, 40)
    b, a = cheby1(N, 0.5, Wn,'bandpass')
    # --------------
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            _x = x[i,:,j]
            x[i,:,j] = filtfilt(b,a,_x,padlen=3*(max(len(b),len(a))-1)) # apply filter
    return x


x, y, __ = prepare_data_as()  # [?,tw,ch]
