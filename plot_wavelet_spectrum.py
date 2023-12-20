import numpy as np
import matplotlib.pyplot as plt
import pywt

# import data
file_type = 3
if file_type == 0:
    file_dir = 'E:/Research/Data/Tianwen/m1a04x_renew/'
    file_name = 'BdBdchan3_1frephase2s.dat'
    var_list = ['Residual Frequency', 'Residual Phase', 'Signal Density', 'Noise Density']
    unit_list = ['Hz', 'rad', 'dB', 'dB']
elif file_type == 1:
    file_dir = 'E:/Research/Data/Tianwen/m1727x/km/amp_fre_phase/TW/'
    file_name = 'TWkmfpda.dat'
    var_list = ['Residual Frequency-1', 'Residual Phase-1', 'Relative Time Delay-1',
                 'Residual Frequency-2', 'Residual Phase-2', 'Relative Time Delay-2',
                 'Signal-to-Noise Ratio', 'Signal Strength']
    unit_list = ['Hz', 'rad', 'ns', 'Hz', 'rad', 'ns', 'dB', 'dB']
elif file_type == 2:
    file_dir = 'E:/Research/Data/Tianwen/m1a04x_renew/'
    file_name = 'Hhchan2freq.dat'
    var_list = ['Residual Frequency-1', 'Residual Frequency-2']
    unit_list = ['Hz', 'Hz']
elif file_type == 3:
    file_dir = 'E:/Research/Data/Tianwen/m1a04x_PLL/'
    file_name = 'HhHhsignum113pll_Ts.dat'
    var_list = ['Residual Frequency-1', 'Residual Phase-1',
                 'Residual Frequency-2', 'Residual Phase-2']
    unit_list = ['Hz', 'Hz']

file_path = file_dir + file_name
data = np.loadtxt(file_path)
# data = data[8000:,:]
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

def plot_all(time, signal, dt, n, fft_freq, fft_values, wave_freq, cwtmatr, cwt_freq, title, var_name, unit_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 5], 'width_ratios': [5, 4]})

    # time series
    axes[0, 0].plot(time, signal)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel(var_name + '[' + unit_name + ']') 
    axes[0, 0].set_xlim(time.min(), time.max())
    axes[0, 0].tick_params(direction='in')
    axes[0, 0].set_title('Time Series')

    # wavelet transform
    pcwt = axes[1, 0].pcolormesh(time, cwt_freq, np.abs(cwtmatr)**2/n, cmap='jet')
    axes[1, 0].set_title('Wavelet Transform')
    axes[1, 0].set_ylabel('Frequency [Hz]')
    axes[1, 0].set_yscale('log')
    cbar = plt.colorbar(pcwt, ax=axes[1, 0], orientation='horizontal', pad=0.1)
    
    # no right-upper subplot
    axes[0, 1].axis('off')
    
    # fourier transform
    axes[1, 1].plot(fft_freq, np.abs(fft_values)**2/n)
    axes[1, 1].set_xlabel('Frequency [Hz]')
    axes[1, 1].set_ylabel('Spectral Density [' + unit_name +'^2/Hz]')
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
    var_name = var_list[i] if i < len(var_list) else f'variable {i + 1}'
    unit_name =  unit_list[i] if i < len(unit_list) else ''
    title = file_path[24:] + ' --- ' + var_name
    
    # eliminate ambiguity for phase sequence
    if file_type == 0 and i == 1:
        signal = eliminate_weekly_ambiguity(signal)

    # calculate FTT
    n = len(time)
    dt = time[1] - time[0]
    print(dt)
    fft_freq = np.fft.fftfreq(n, dt)
    fft_values = np.fft.fft(signal)

    # calculate CWT
    wavename = 'cmorl1.5-1.0'
    num_freq = 50
    wave_freq = np.logspace(-4, np.log10(1/(2*dt)), num_freq)
    if file_type == 1:
        wave_freq = np.logspace(-1, 0.8*np.log10(1/(2*dt)), num_freq)
    scales = 1 / wave_freq
    cwtmatr, cwt_freq = pywt.cwt(signal, scales, wavename, 1.0 / dt)
    # print(cwt_freq)

    # plot figure
    plot_all(time, signal, dt, n, fft_freq[fft_freq > 0], fft_values[fft_freq > 0], wave_freq, cwtmatr, cwt_freq, title, var_name, unit_name)
