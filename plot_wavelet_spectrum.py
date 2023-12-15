import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pywt

# import data
file_type = 0
if file_type == 0:
    file_dir = 'E:/Research/Data/Tianwen/m1a04x_renew/'
    file_name = 'HhHhchan3_1frephase2s.dat'
    var_list = ['Residual Frequency [Hz]', 'Residual Phase [rad]', 'Signal Density [dB]', 'Noise Density [dB]']
elif file_type == 1:
    file_dir = 'E:/Research/Data/Tianwen/m1727x/km/amp_fre_phase/TW/'
    file_name = 'TWkmfpda.dat'
    var_list = ['Residual Frequency-1 [Hz]', 'Residual Phase-1 [rad]', 'Relative Time Delay-1 [ns]',
                 'Residual Frequency-2 [Hz]', 'Residual Phase-2 [rad]', 'Relative Time Delay-2 [ns]',
                 'Signal-to-Noise Ratio [dB]', 'Signal Strength [dB]']
if file_type == 2:
    file_dir = 'E:/Research/Data/Tianwen/m1a04x_renew/'
    file_name = 'Hhchan2freq.dat'
    var_list = ['Residual Frequency-1 [Hz]', 'Residual Frequency-2 [Hz]']

file_path = file_dir + file_name
data = np.loadtxt(file_path)
# data = data[300:,:]
time = data[:, 0]
var = data[:, 1:]
if file_type == 2:
    time = data[:, 1]
    var = data[:, 2:]
time = time - time [0]
if file_type == 0:
    time = np.linspace(0,int(file_name[19])*(len(time)-1),len(time))
if file_type == 1:
    var = var[:, 1:]

def plot_all(time, signal, fft_freq, fft_values, wave_freq, cwtmatr, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 5], 'width_ratios': [5, 4]})

    # time series
    axes[0, 0].plot(time, signal)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_xlim(time.min(), time.max())
    axes[0, 0].tick_params(direction='in')
    axes[0, 0].set_title('Time Series')

    # wavelet transform
    im = axes[1, 0].imshow(np.abs(cwtmatr), aspect='auto', extent=[time[0], time[-1], wave_freq[0], wave_freq[-1]], cmap='jet', interpolation='bilinear')
    axes[1, 0].set_title('Wavelet Transform')
    axes[1, 0].set_ylabel('Frequency [Hz]')
    axes[1, 0].set_yscale('log')
    cbar = plt.colorbar(im, ax=axes[1, 0], orientation='horizontal', pad=0.1)
    
    # no right-upper subplot
    axes[0, 1].axis('off')
    
    # fourier transform
    axes[1, 1].plot(fft_freq, np.abs(fft_values))
    axes[1, 1].set_xlabel('Frequency [Hz]')
    axes[1, 1].set_ylabel('Amplitude [dB]')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].tick_params(direction='in')
    axes[1, 1].set_title('Fourier Transform')

    fig.suptitle(title)
    plt.show()

def eliminate_weekly_ambiguity(phase_sequence):
    for i in range(1, len(phase_sequence)):
        if phase_sequence[i] > phase_sequence[i - 1]:
            phase_sequence[i:] -= 2 * np.pi
    return phase_sequence

for i in range(var.shape[1]):
    signal = var[:, i]
    var_title = var_list[i] if i < len(var_list) else f'variable {i + 1}'
    title = file_path[24:] + ' --- ' + var_title
    
    # eliminate ambiguity for phase sequence
    if file_type == 0 and i == 1:
        signal = eliminate_weekly_ambiguity(signal)

    # calculate FTT
    n = len(time)
    dt = time[1] - time[0]
    print(dt)
    fft_freq = np.fft.fftfreq(n, dt)
    fft_values = fft(signal)

    # calculate CWT
    wavename = 'cmorl1.5-1.0'
    num_freq = 50
    wave_freq = np.logspace(-4, np.log10(1/(2*dt)), num_freq)
    if file_type == 1:
        wave_freq = np.logspace(-1, 0.8*np.log10(1/(2*dt)), num_freq)
    scales = 1 / wave_freq
    cwtmatr, _ = pywt.cwt(signal, scales, wavename, 1.0 / dt)

    # plot figure
    plot_all(time, signal, fft_freq[fft_freq > 0], fft_values[fft_freq > 0], wave_freq, cwtmatr, title)
