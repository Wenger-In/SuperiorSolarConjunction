import matplotlib.pyplot as plt
import pandas as pd
import pyleoclim as pyleo
import numpy as np
from matplotlib.colors import SymLogNorm, LogNorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
from scipy.interpolate import interp1d, CubicSpline
import sys

sys.path.append(r'E:/Research/Program/else')
import python_utils_JSHEPT
from python_utils_JSHEPT import get_plot_WaveletAnalysis_of_var_vect, wavelet_reconstruction

sys.path.append(r'E:/Research/Program/SuperiorSolarConjunction')
import analyse_wavelet_coherence
from analyse_wavelet_coherence import convert_to_second_of_day, eliminate_outliers, interpolate, detrend

def log_linear_fit(freq, psd):
    log_freq = np.log10(freq)
    log_psd = np.log10(psd)
    coef = np.polyfit(log_freq, log_psd, 1)
    slope, intercept = coef[0], coef[1]
    log_psd_fit = np.polyval(coef, log_freq)
    psd_fit = 10**log_psd_fit
    return slope, psd_fit

## import data
i_case = 2
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
    file1_name = file_Js
    file2_name = file_Bd
    time_beg = 2021277080000
    time_end = 2021277083000
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
sod_beg = convert_to_second_of_day(time_beg)[0]
sod_end = convert_to_second_of_day(time_end)[0]
    
## read data
file1_path = file_dir + file1_name
file2_path = file_dir + file2_name
data1 = np.loadtxt(file1_path)
data2 = np.loadtxt(file2_path)

## extrcat data
time1 = data1[:, 0]
time2 = data2[:, 0]
sod1 = convert_to_second_of_day(time1)
sod2 = convert_to_second_of_day(time2)
if file1_name == 'JsJschan3_1frephase1s.dat':
    sod1 = time1 # time has been formatted as 'seconds of day'
freq1 = data1[:,1]
freq2 = data2[:,1]

## select time interval
ind1_sub = np.where((sod1 > sod_beg) & (sod1 < sod_end))
ind2_sub = np.where((sod2 > sod_beg) & (sod2 < sod_end))
# extract corresponding segment
sod1_sub = sod1[ind1_sub]
sod2_sub = sod2[ind2_sub]
freq1_sub = freq1[ind1_sub]
freq2_sub = freq2[ind2_sub]

## frequency series preprocess
# step 1: eliminate outliers
freq1_out, sod1_out = eliminate_outliers(freq1_sub, sod1_sub, 2)
freq2_out, sod2_out = eliminate_outliers(freq2_sub, sod2_sub, 2)
# step 2: detrend for frequency sequence
freq1_fit, freq1_detrend = detrend(freq1_out, sod1_out, 3)
freq1_fit, freq2_detrend = detrend(freq2_out, sod2_out, 3)
# step 3: interpolation for frequency sequence
freq1_interp = interpolate(freq1_detrend, sod1_out, sod1_sub)
freq2_interp = interpolate(freq2_detrend, sod2_out, sod2_sub)

time1_vect, period1_vect, WaveletObj_arr1, WaveletCoeff_arr1, sub_wave_arr1 = \
    get_plot_WaveletAnalysis_of_var_vect(sod1_sub, freq1_interp, period_range=np.array([2,500]), num_periods=100)
WaveletCoeff_arr1 = np.transpose(WaveletCoeff_arr1)
time2_vect, period2_vect, WaveletObj_arr2, WaveletCoeff_arr2, sub_wave_arr2 = \
    get_plot_WaveletAnalysis_of_var_vect(sod2_sub, freq2_interp, period_range=np.array([2,500]), num_periods=100)
WaveletCoeff_arr2 = np.transpose(WaveletCoeff_arr2)

timee1, periodd1 = np.meshgrid(time1_vect, period1_vect)
timee2, periodd2 = np.meshgrid(time2_vect, period2_vect)

# ## plot wavelet coefficient spectrum
# plt.figure()
# plt.subplot(2,1,1)
# plt.pcolormesh(timee1, periodd1, WaveletCoeff_arr1.real)
# plt.yscale('log')
# plt.xlabel('Time[s]')
# plt.ylabel('Scale[s]')
# plt.subplot(2,1,2)
# plt.pcolormesh(timee2, periodd2, WaveletCoeff_arr2.real)
# plt.yscale('log')
# plt.xlabel('Time[s]')
# plt.ylabel('Scale[s]')
# plt.close()

##### Part I: reconstruct wavelet
## extract wavelet component
period_lb, period_ub = 162, 230
period1_slc_ind = np.where((period1_vect > period_lb) & (period1_vect < period_ub))[0]
period2_slc_ind = np.where((period2_vect > period_lb) & (period2_vect < period_ub))[0]
period1_slc = period1_vect[period1_slc_ind]
period2_slc = period2_vect[period2_slc_ind]
sub_wave1_slc = sub_wave_arr1[:,period1_slc_ind]
sub_wave2_slc = sub_wave_arr2[:,period2_slc_ind]

## plotly figure
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.02)

# panel 1: time series
fig.add_trace(go.Scatter(x=sod1_sub, y=freq1_interp, mode='lines', marker={'color':'blue'}, name=file1_name[0:2]), \
    row=1, col=1)
fig.add_trace(go.Scatter(x=sod2_sub, y=freq2_interp, mode='lines', marker={'color':'red'}, name=file2_name[0:2]), \
    row=1, col=1)
fig.update_yaxes(title='Freq [Hz]', row=1, col=1)
# panel 2: reconstructed series
fig.add_trace(go.Scatter(x=sod1_sub, y=np.sum(sub_wave1_slc, axis=1), mode='lines', marker={'color':'blue'}, \
    name=file1_name[0:2]+'-recon'), row=2, col=1)
fig.add_trace(go.Scatter(x=sod2_sub, y=np.sum(sub_wave2_slc, axis=1), mode='lines', marker={'color':'red'}, \
    name=file2_name[0:2]+'-recon'), row=2, col=1)
fig.update_yaxes(title='Freq [Hz]', row=2, col=1)
# figure layout
fig.update_xaxes(title_text='Time [s]', row=2, col=1)
fig.update_layout(title={'text': 'Wavelet Reconstruction','x': 0.5, 'y': 0.95})

###### Part II: calculate power spectrum density index
## calculate wavelet power spectrum
psd1 = np.abs(WaveletCoeff_arr1)**2 * period1_vect.reshape(-1,1)
psd2 = np.abs(WaveletCoeff_arr2)**2 * period2_vect.reshape(-1,1)

## select time interval for calculating power index
sod_lb, sod_ub = 29200, 29400
sod1_slc_ind = np.where((time1_vect > sod_lb) & (time1_vect < sod_ub))[0]
sod2_slc_ind = np.where((time2_vect > sod_lb) & (time2_vect < sod_ub))[0]
sod1_slc = time1_vect[sod1_slc_ind]
sod2_slc = time2_vect[sod2_slc_ind]
psd1_slc = psd1[:, sod1_slc_ind]
psd2_slc = psd2[:, sod2_slc_ind]

## calculate power index
freq1_vect = 1/period1_vect
freq2_vect = 1/period2_vect
psd1_mean = np.mean(psd1_slc, 1)
psd2_mean = np.mean(psd2_slc, 1)

psd_slope1, psd_fit1 = log_linear_fit(freq1_vect, psd1_mean)
psd_slope2, psd_fit2 = log_linear_fit(freq2_vect, psd2_mean)

## plot figure
plt.figure(figsize=(10, 6))
plt.subplot2grid((2, 3), (0, 0), colspan=2)
plt.pcolormesh(timee1, periodd1, psd1, norm=LogNorm())
plt.clim(np.min(psd1)*((np.max(psd1)/np.min(psd1))**(1/4)), np.max(psd1))
plt.colorbar()
plt.yscale('log')
plt.ylabel('Scale[s]')

plt.subplot2grid((2, 3), (0, 2))
plt.plot(freq1_vect, psd1_mean)
plt.plot(freq1_vect, psd_fit1, label='alpha=' + str(-round(psd_slope1,2)))
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.ylabel(file1_name[0:2] + '-PSD[Hz2/Hz]')

plt.subplot2grid((2, 3), (1, 0), colspan=2)
plt.pcolormesh(timee2, periodd2, psd2, norm=LogNorm())
plt.clim(np.min(psd2)*((np.max(psd2)/np.min(psd2))**(1/4)), np.max(psd2))
plt.colorbar()
plt.yscale('log')
plt.xlabel('Time[s]')
plt.ylabel('Scale[s]')

plt.subplot2grid((2, 3), (1, 2))
plt.plot(freq2_vect, psd2_mean)
plt.plot(freq2_vect, psd_fit2, label='alpha=' + str(-round(psd_slope2,2)))
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Freq[Hz]')
plt.ylabel(file2_name[0:2] + '-PSD[Hz2/Hz]')

fig.show()

plt.show()

db
