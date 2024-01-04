import numpy as np
from scipy import integrate
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

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

def cal_sigma_Ne(sigma_FM, nu_c, L_LOS, r):
    re = 2.8179e-15 # electron radius, [m]
    Rs = 6.955e8 # solar radius, [m]
    R = r*Rs
    sigma_Ne = sigma_FM / re / nu_c / np.sqrt(L_LOS*R)
    print('sigma_Ne: ', round(sigma_Ne,3))
    return sigma_Ne

def cal_Ne(r):
    Ne = (30/r**6 + 1/r**2.2) * 1e12 # model
    return Ne

def cal_L_LOS(r):
    L_LOS = 3.35e6*r**0.918
    return L_LOS

def cal_sigma_FM(FM, freq, nu_a, nu_b):
    FM2 = FM ** 2
    def integrand(x):
        return np.interp(x, freq, FM2)
    integrate_result = integrate.quad(integrand, nu_a, nu_b)
    sigma_FM = np.sqrt(integrate_result[0])
    print('sigma_FM: ', round(sigma_FM,3))
    return sigma_FM

def cal_nu_c(beta, nu_a, nu_b):
    nu_c = np.sqrt((beta+1)/(beta-1) * (nu_b**(1-beta)-nu_a**(1-beta))/(nu_b**(-1-beta)-nu_a**(-1-beta)))
    print('nu_c: ', round(nu_c,3))
    return nu_c

def cal_FM(FF2, freq_band):
    c = 3e8
    lambda_band = c / freq_band
    FM = np.sqrt(FF2) / lambda_band
    return FM

def cal_beta(FF2, freq, nu_a, nu_b):
    indices_sub = (freq >= nu_a) & (freq <= nu_b)
    freq_sub = freq[indices_sub]
    FF2_sub = FF2[indices_sub]
    fit_coef = np.polyfit(np.log10(freq_sub), np.log10(FF2_sub), deg=1)
    FF2_log10_fit = np.polyval(fit_coef, np.log10(freq_sub))
    FF2_fit = np.power(10, FF2_log10_fit)
    slope = round(fit_coef[0],3)
    beta = -slope
    print('beta: ', beta)
    plt.figure()
    plt.plot(fft_freq, FF2)
    plt.plot(freq_sub, FF2_fit)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Spectral Density [' + unit_name +'^2/Hz]')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Frequency Fluctuation PSD')
    plt.annotate(f'slope={slope}', xy=(1e-3,1e-3), xytext=(1e-3,1e-3))
    plt.show()
    return beta

file_dir = 'E:/Research/Data/Tianwen/m1a04x_PLL/'
file_name = 'BdBdsignum113pll_Ts.dat'
var_list = ['Residual Frequency-1', 'Residual Phase-1', 'Residual Frequency-2', 'Residual Phase-2']
unit_list = ['Hz', 'rad', 'Hz', 'rad']

file_path = file_dir + file_name
data = np.loadtxt(file_path)
# data = data[8000:,:] # for HhHhsignum113pll_Ts.dat
time = data[:, 0]
var = data[:, 1:]
time = time - time[0]

for i in range(var.shape[1]):
    if i == 1 or i == 3: # not for phase sequence
        continue
    signal = var[:, i]
    var_name = var_list[i] if i < len(var_list) else f'variable {i + 1}'
    unit_name =  unit_list[i] if i < len(unit_list) else ''
    title = file_path[24:] + ' --- ' + var_name
    
    # linear detrend for phase sequence
    if i == 1 or i == 3:
        signal = linear_detrend(signal, time, title)
    
    # quadratic detrend for frequency sequence
    if i == 0 or i == 2:
        signal = quadratic_detrend(signal, time, title)

    # calculate FTT
    n = len(time)
    dt = time[1] - time[0]
    fs = 1/dt
    print(round(dt,3))
    fft_freq = np.fft.fftfreq(n, dt)
    fft_amp = np.fft.fft(signal)
    fft_amp = fft_amp[fft_freq > 0]
    FF2 = np.abs(fft_amp) ** 2 / (n/fs)
    fft_freq = fft_freq[fft_freq > 0]
    
    # calculate spectral index beta
    nu_Nyq = 1/(2*dt)
    nu_a = 1e-3
    nu_b = 0.01 * nu_Nyq
    beta = cal_beta(FF2, fft_freq, nu_a, nu_b)
    
    # calculate scaling frequency factor
    nu_c = cal_nu_c(beta, nu_a, nu_b)
    
    # calcualte fluctuation measure FM
    freq_band = 8.4e9
    FM = cal_FM(FF2, freq_band)
    plt.figure()
    plt.plot(fft_freq, FM**2)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Flctuation Measure [' + unit_name +'^2/Hz]')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Fluctuation Measure PSD')
    plt.show()
    
    # calculate Ne and L_LOS based on models
    r = 2
    Ne = cal_Ne(r)
    L_LOS = cal_L_LOS(r)

    # calculate sigma_FM and sigma_Ne
    sigma_FM = cal_sigma_FM(FM, fft_freq, nu_a, nu_b)
    sigma_Ne = cal_sigma_Ne(sigma_FM, nu_c, L_LOS, r)
    
    # calculate fractional density fluctuations
    epsilon = sigma_Ne / Ne
    print('epsilon: ', round(epsilon,3))