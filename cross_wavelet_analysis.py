import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import pywt
from matplotlib.colors import LogNorm
from matplotlib.image import NonUniformImage

file_dir = 'E:/Research/Data/Tianwen/m1a04x_PLL/'
file1_name = 'BdBdsignum113pll_Ts.dat'
file2_name = 'HhHhsignum113pll_Ts.dat'
var_list = ['Residual Frequency-1', 'Residual Phase-1', 'Residual Frequency-2', 'Residual Phase-2']
unit_list = ['Hz', 'rad', 'Hz', 'rad']

file1_path = file_dir + file1_name
file2_path = file_dir + file2_name
data1 = np.loadtxt(file1_path)
data2 = np.loadtxt(file2_path)
slt_indices = np.arange(10000,30000)
time = data1[slt_indices, 0]
var1 = data1[slt_indices, 1:]
var2 = data2[slt_indices, 1:]

def linear_detrend(phase_sequence, time, title):
    coefficients = np.polyfit(time, phase_sequence, 1)
    fit_sequence = np.polyval(coefficients, time)
    detrended_sequence = phase_sequence - fit_sequence
    plt.figure()
    plt.plot(time,phase_sequence,label='phase')
    plt.plot(time,fit_sequence,label='fit')
    plt.legend()
    plt.title(title)
    plt.show()
    return detrended_sequence

def quadratic_detrend(frequency_sequence,time,title):
    coefficients = np.polyfit(time, frequency_sequence, 2)
    fit_sequence = np.polyval(coefficients, time)
    detrended_sequence = frequency_sequence - fit_sequence
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    axs[0].plot(time, frequency_sequence)
    axs[0].plot(time, fit_sequence)
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].legend()
    axs[1].plot(time, detrended_sequence)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Detrended Frequency [Hz]')
    plt.suptitle(title)
    plt.show()
    return detrended_sequence

for i in range(var1.shape[1]):
    signal1 = var1[:, i]
    signal2 = var2[:, i]
    var_name = var_list[i] if i < len(var_list) else f'variable {i + 1}'
    unit_name =  unit_list[i] if i < len(unit_list) else ''
    title = var_name
    
    # quadratic detrend for frequency sequence
    if i == 0 or i == 2:
        signal1 = quadratic_detrend(signal1, time, title)
        signal2 = quadratic_detrend(signal2, time, title)

    # linear detrend for phase sequence
    if i == 1 or i == 3:
        signal1 = linear_detrend(signal1, time, title)
        signal2 = linear_detrend(signal2, time, title)

    # calculate CWT
    n = len(time)
    dt = time[1] - time[0]
    fs = 1/dt
    print(dt)
    # wavename = 'cmorl1.5-1.0'
    wavename = 'cmor'
    num_freq = 50
    wave_freq = np.logspace(-3, np.log10(1/(2*dt)), num_freq)
    scales = 1 / wave_freq
    cwtmatr1, cwt_freq = pywt.cwt(signal1, scales, wavename, 1.0 / dt)
    cwtmatr2, cwt_freq = pywt.cwt(signal2, scales, wavename, 1.0 / dt)
    
    # calculate cross wavelet spectrum
    cross_wavelet = cwtmatr1 * np.conj(cwtmatr2)
    
    # calculate wavelet coherence phase
    real_part = np.real(cross_wavelet)
    imag_part = np.imag(cross_wavelet)
    cohe_phase = np.angle(cross_wavelet)
    
    # calculate wavelet lag
    wave_freq_2d = wave_freq[:, np.newaxis]
    wavelet_lag = cohe_phase / wave_freq_2d

    # plot figure
    plt.figure(figsize=(16, 6))
    # wavelet spectra for Bd
    plt.subplot(2, 4, 1)
    plt.pcolormesh(time, wave_freq, np.abs(cwtmatr1), cmap='jet', shading='auto')
    plt.colorbar(orientation='horizontal', pad=0.1)
    plt.title('Bd wavelet')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.yscale('log')
    # wavelet spectra for Hh
    plt.subplot(2, 4, 2)
    plt.pcolormesh(time, wave_freq, np.abs(cwtmatr2), cmap='jet', shading='auto')
    plt.colorbar(orientation='horizontal', pad=0.1)
    plt.title('Hh wavelet')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.yscale('log')
    # cross wavelet spectra
    plt.subplot(2, 4, 3)
    plt.pcolormesh(time, wave_freq, np.abs(cross_wavelet), cmap='jet', shading='auto')
    plt.colorbar(orientation='horizontal', pad=0.1)
    plt.title('cross wavelet spectra')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.yscale('log')
    # real part (co-wavelet spectra)
    plt.subplot(2, 4, 5)
    plt.pcolormesh(time, wave_freq, real_part, cmap='RdBu', shading='auto')
    plt.colorbar(orientation='horizontal', pad=0.1)
    plt.title('real part (co-wavelet spectra)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.yscale('log')
    # imaginary part (quadrature-wavelet spectra)
    plt.subplot(2, 4, 6)
    plt.pcolormesh(time, wave_freq, imag_part, cmap='RdBu', shading='auto')
    plt.colorbar(orientation='horizontal', pad=0.1)
    plt.title('imaginary part (quadrature-wavelet spectra)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.yscale('log')
    # coherence angle
    plt.subplot(2, 4, 7)
    plt.pcolormesh(time, wave_freq, np.degrees(cohe_phase), cmap='RdBu', shading='auto')
    plt.colorbar(orientation='horizontal', pad=0.1)
    plt.title('coherence angle')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.yscale('log')
    # wavelet lag
    plt.subplot(2, 4, 8)
    plt.pcolormesh(time, wave_freq, wavelet_lag, cmap='RdBu', shading='auto')
    plt.colorbar(orientation='horizontal', pad=0.1)
    plt.title('wavelet lag')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.yscale('log')

    plt.tight_layout()
    plt.show()