import matplotlib.pyplot as plt
import pandas as pd
import pyleoclim as pyleo
import numpy as np
from matplotlib.colors import SymLogNorm
from scipy.stats import zscore
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d, CubicSpline

import sys
sys.path.append(r'E:/Research/Program/else')
import python_utils_JSHEPT
from python_utils_JSHEPT import get_plot_WaveletAnalysis_of_var_vect, wavelet_reconstruction

## function: convert the time data to 'seconds of day'
def convert_to_second_of_day(time_array):
    if isinstance(time_array, int):
        time_array = [time_array]
    sod_array = []
    for time in time_array:
        # time = int(time)
        year = time // 1000000000
        doy = (time % 1000000000) // 1000000
        hour = (time % 1000000) // 10000
        minute = (time % 10000) // 100
        second = time % 100
        sod = hour * 3600 + minute * 60 + second
        sod_array.append(sod)
    return np.array(sod_array)

## function: eliminate outliers with deviation > threshold*std_error
def eliminate_outliers(freq, t, threshold):
    freq_zscore = zscore(freq)
    outliers = np.abs(freq_zscore) > threshold
    outliers[0] = False
    outliers[-1] = False
    freq = freq[~outliers]
    t = t[~outliers]
    return freq, t

# function: interpolate frequency time series
def interpolate(freq, t, t_std, method='linear'):
    if method == 'linear':
        f = interp1d(t, freq)
    elif method == 'quad':
        f = interp1d(t, freq, kind='quadratic')
    elif method == 'cubic':
        f = CubicSpline(t, freq)
    freq_std = f(t_std)
    return freq_std

## function: detrend frequency time series and plot it
def detrend(freq, t, n, title):
    coef = np.polyfit(t, freq, n)
    freq_fit = np.polyval(coef, t)
    freq_detrended = freq - freq_fit
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, freq)
    axs[0].plot(t, freq_fit)
    axs[0].set_ylabel('Freq [Hz]')
    axs[0].legend()
    axs[1].plot(t, freq_detrended)
    axs[1].set_xlabel('Second of Day [s]')
    axs[1].set_ylabel('Freq Detrended [Hz]')
    plt.suptitle(title)
    # plt.show()
    plt.close()
    return freq_detrended

## import data
i_case = 4
save_or_not = 0
if i_case == 1: # 2021/09/30(273), 12:00-13:00, Ht-Sv, Ht-Wz, Sv-Wz, (latitudinal fluctuaion)
    file_dir = 'E:/Research/Data/Tianwen/m1930x_up_new/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1930x_up_new/'
    file_Ht = 'HtHtchan3_1frephase1s.dat'
    file_Sv = 'SvSvchan3_1frephase1s.dat'
    file_Wz = 'WzWzchan3_1frephase1s.dat'
    file1_name = file_Ht
    file2_name = file_Sv
    time_beg = 2021273120000
    time_end = 2021273123000
elif i_case == 2: # 2021/10/04(277), 05:40-08:20, Js-Bd, Bd-Yg, Yg-Hh, (inward propagation, latitudinal fluctuaion, outward propagation)
    file_dir = 'E:/Research/Data/Tianwen/m1a04x_renew/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a04x_renew/'
    file_Js = 'JsJschan3_1frephase1s.dat' # time has been formatted as 'seconds of day'
    file_Bd = 'BdBdchan3_1frephase1s.dat'
    file_Yg = 'YgYgchan3_1frephase2s.dat'
    file_Hh = 'HhHhchan3_1frephase1s.dat'
    file1_name = file_Yg
    file2_name = file_Hh
    time_beg = 2021277053000
    time_end = 2021277060000
elif i_case == 3: # 2021/10/07(280), 03:30-04:00, sh-km, (polar region fluctuation)
    file_dir = 'E:/Research/Data/Tianwen/m1a07x_up/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a07x_up/'
    file_sh = 'shshchan3_1frephase5s.dat'
    file_km = 'kmkmchan3_1frephase5s.dat'
    file1_name = file_sh
    file2_name = file_km
    time_beg = 2021280033000
    time_end = 2021280040000
elif i_case == 4: # 2021/10/15(288), 01:00-04:00, Bd-Hh, Bd-Ys, Hh-Ys, (fine structure)
                  #                  07:40-13:00, Bd-Hh, Bd-Ys, Hh-Ys, (CME)
    file_dir = 'E:/Research/Data/Tianwen/m1a15y_copy/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a15y_copy/'
    file_Bd = 'BdBdchan3_1frephase1s.dat'
    file_Hh = 'HhHhchan3_1frephase1s.dat'
    file_Ys = 'YsYschan3_1frephase1s.dat'
    file1_name = file_Bd
    file2_name = file_Hh
    time_beg = 2021288073000
    time_end = 2021288080000
# convert subplot time interval
str_beg = str(time_beg)[-6:-2]
str_end = str(time_end)[-6:-2]
sub_beg = convert_to_second_of_day(time_beg)[0]
sub_end = convert_to_second_of_day(time_end)[0]
    
## read data
file1_path = file_dir + file1_name
file2_path = file_dir + file2_name
data1 = np.loadtxt(file1_path)
data2 = np.loadtxt(file2_path)

## extrcat data
# time
time1 = data1[:, 0]
time2 = data2[:, 0]
sod1 = convert_to_second_of_day(time1)
sod2 = convert_to_second_of_day(time2)
if file1_name == 'JsJschan3_1frephase1s.dat':
    sod1 = time1 # time has been formatted as 'seconds of day'
# frequency
freq1 = data1[:,1]
freq2 = data2[:,1]

## select steady interval
# steady indices
ind1_sub = np.where((sod1 > sub_beg) & (sod1 < sub_end))
ind2_sub = np.where((sod2 > sub_beg) & (sod2 < sub_end))
# extract corresponding segment
sod1_sub = sod1[ind1_sub]
sod2_sub = sod2[ind2_sub]
freq1_sub = freq1[ind1_sub]
freq2_sub = freq2[ind2_sub]

# eliminate outliers
freq1_out, sod1_out = eliminate_outliers(freq1_sub, sod1_sub, 2)
freq2_out, sod2_out = eliminate_outliers(freq2_sub, sod2_sub, 2)

## detrend for frequency sequence
freq1_out = detrend(freq1_out, sod1_out, 3, file1_name + '-Freq[Hz]')
freq2_out = detrend(freq2_out, sod2_out, 3, file2_name + '-Freq[Hz]')

## interpolation for frequency sequence
freq1_sub = interpolate(freq1_out, sod1_out, sod1_sub)
freq2_sub = interpolate(freq2_out, sod2_out, sod2_sub)

time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = \
    get_plot_WaveletAnalysis_of_var_vect(sod1_sub, freq1_sub, period_range=np.array([2,500]), num_periods=100)
WaveletCoeff_var_arr = np.transpose(WaveletCoeff_var_arr)

## plot wavelet coefficient spectrum
plt.figure()
timee, periodd = np.meshgrid(time_vect, period_vect)
plt.pcolor(timee, periodd, WaveletCoeff_var_arr.real)
plt.yscale('log')

## extract wavelet component
period_lb, period_ub = 14, 25
period_slc_ind = np.where((period_vect > period_lb) & (period_vect < period_ub))[0]
period_slc = period_vect[period_slc_ind]
sub_wave_var_slc = sub_wave_var_arr[:,period_slc_ind]

plt.figure()
plt.subplot(3,1,1)
plt.plot(sod1_sub, freq1_sub)
plt.subplot(3,1,2)
plt.plot(sod1_sub, np.sum(sub_wave_var_arr, axis=1))
plt.subplot(3,1,3)
plt.plot(sod1_sub, np.sum(sub_wave_var_slc, axis=1))

## calculate wavelet power spectrum
WaveletPower_var_arr = np.abs(WaveletCoeff_var_arr)**2 * period_vect.reshape(-1,1)

plt.figure()
plt.pcolor(timee, periodd, WaveletPower_var_arr)
plt.colorbar()
plt.yscale('log')

## select time interval for calculating power index
sod_lb, sod_ub = 27900, 28000
sod_slc_ind = np.where((time_vect > sod_lb) & (time_vect < sod_ub))[0]
sod_slc = time_vect[sod_slc_ind]
WaveletPower_var_slc = WaveletPower_var_arr[:, sod_slc_ind]

## calculate power index
freq_vect = 1/period_vect
ps_mean = np.mean(WaveletPower_var_slc, 1)

plt.figure()
plt.plot(freq_vect, ps_mean)
plt.xscale('log')
plt.yscale('log')

plt.show()

db
