import numpy as np
import random

x = np.linspace(0, 2 * np.pi, 1000)
y_sin = np.sin(x)
y_cos = np.cos(x)

x = np.array([],dtype=np.float32).reshape(0,9,125) # data
y = np.array([],dtype=np.float32).reshape(0) # label
i = 0
for n_samples in range(50):
    for f0 in np.arange(8,15,0.5):
        A = 1
        fs = 250
        phi = 0
        t = 0.5

        T = 1.0 / fs
        N = t / T

        n = np.arange(N)  # [0,1,..., N-1]
        _x = np.zeros([1,9,125], dtype=np.float32)
        _y = np.zeros(1, dtype=np.float32)
        for channel in range(9):
            _x[0,channel,:] = A * np.sin(2 * f0 * (np.pi) * n * T + (phi+np.pi/4*channel))
        _y = f0
        x = np.append(x,_x,axis=0)
        y = np.append(y,_y)
        i = i+1
print(x.shape)

for x1 in range(x.shape[0]):
    for x2 in range(x.shape[1]):
        for x3 in range(x.shape[2]):
            x[x1,x2,x3] = x[x1,x2,x3] + random.gauss(0,0.5)

print(x.shape)

dat = x
lab = y
save_pathx = "E:/Devin/djl/异步论文离线实验数据/AllData/npy/S" + str(0 + 1) + "/" + "ssvep-test.npy"
np.save(save_pathx, dat)
save_pathy = "E:/Devin/djl/异步论文离线实验数据/AllData/npy/S" + str(0 + 1) + "/" + "ssvep-test-label.npy"
np.save(save_pathy, lab)