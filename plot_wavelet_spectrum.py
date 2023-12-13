import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pywt

# 读取数据文件
data = np.loadtxt('E:/Research/Data/Tianwen/m1930x_up_new/HtHtchan3_1frephase2s.dat')

# 提取时间列和物理量列
time = data[:, 0]
physical_quantities = data[:, 1:]

# 定义傅里叶变换和小波变换函数
def plot_fft(time, signal, title):
    n = len(time)
    dt = time[1] - time[0]
    freq = np.fft.fftfreq(n, dt)
    fft_values = fft(signal)
    plt.figure(figsize=(10, 4))
    plt.plot(freq, np.abs(fft_values))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_wavelet_transform(time, signal, title):
    scales = np.arange(1, len(time) + 1)
    coefs, freqs = pywt.cwt(signal, scales, 'morl')
    plt.figure(figsize=(10, 4))
    plt.imshow(np.abs(coefs), aspect='auto', extent=[time[0], time[-1], 1, len(scales)], cmap='jet', interpolation='bilinear')
    plt.colorbar(label='Magnitude')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.show()

# 绘制原始物理量的时间序列图
for i in range(physical_quantities.shape[1]):
    signal = physical_quantities[:, i]
    title = f'Physical Quantity {i + 1} - Time Series'
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


# 对每一列物理量进行傅里叶变换和小波变换
for i in range(physical_quantities.shape[1]):
    signal = physical_quantities[:, i]
    title_fft = f'FFT - Physical Quantity {i + 1}'
    title_wavelet = f'Wavelet Transform - Physical Quantity {i + 1}'

    # 绘制傅里叶变换谱
    plot_fft(time, signal, title_fft)

    # 绘制小波变换谱
    # plot_wavelet_transform(time, signal, title_wavelet)
