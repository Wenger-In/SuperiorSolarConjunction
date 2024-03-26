import matplotlib.pyplot as plt
import pandas as pd
import pyleoclim as pyleo
import numpy as np
from matplotlib.colors import SymLogNorm
from scipy.stats import zscore
from scipy.interpolate import interp1d, CubicSpline
import plotly.graph_objects as go
from plotly.subplots import make_subplots

## function: convert the time data to 'seconds of day'
def convert_to_second_of_day(time_array):
    if isinstance(time_array, int):
        time_array = [time_array]
    sod_array = []
    for time in time_array:
        year = time // 1000000000
        doy = (time % 1000000000) // 1000000
        hour = (time % 1000000) // 10000
        minute = (time % 10000) // 100
        second = time % 100
        sod = hour * 3600 + minute * 60 + second
        sod_array.append(sod)
    return np.array(sod_array)

## function: eliminate outliers with deviation > threshold*std_error
def eliminate_outliers(freq, time, threshold):
    freq_zscore = zscore(freq)
    outliers = np.abs(freq_zscore) > threshold
    outliers[0] = False
    outliers[-1] = False # retain beginning/end data for interpolation
    freq = freq[~outliers]
    time = time[~outliers]
    return freq, time

# function: interpolate frequency time series
def interpolate(freq, time, time_std, method='linear'):
    if method == 'linear':
        f = interp1d(time, freq)
    elif method == 'quad':
        f = interp1d(time, freq, kind='quadratic')
    elif method == 'cubic':
        f = CubicSpline(time, freq)
    freq_std = f(time_std)
    return freq_std

## function: detrend frequency time series and plot it
def detrend(freq, time, order):
    coef = np.polyfit(time, freq, order)
    freq_fit = np.polyval(coef, time)
    freq_detrended = freq - freq_fit
    return freq_fit, freq_detrended

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
    file1_name = file_Hh
    file2_name = file_Ys
    time_beg = 2021288123000
    time_end = 2021288130000
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

## construct pyleo.series         
series1 = pyleo.Series(time=sod1_sub, value=freq1_interp, \
    time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file1_name[0:2])
series2 = pyleo.Series(time=sod2_sub, value=freq2_interp, \
    time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file2_name[0:2])

# ## transform analysis
# # Fourier transform
# fft1 = series1.spectral(method='wwz')
# fft2 = series2.spectral(method='wwz')
# # wavelet transform
# cwt1 = series1.wavelet(method='wwz')
# cwt2 = series2.wavelet(method='wwz')

## wavelet coherence analysis
coh = series2.wavelet_coherence(series1, method='wwz')
coh.wtc[coh.wtc>1] = np.nan
time_lag = np.flipud(np.rot90(coh.phase/2/np.pi/coh.frequency[np.newaxis,:]))
scale_range = [np.log10(np.min(coh.scale)), np.log10(np.max(coh.scale))]

## plot coherence and time lag spectrum
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.02)

# panel 1: time series
fig.add_trace(go.Scatter(x=sod1_sub, y=freq1_interp, mode='lines', name=file1_name[0:2]), row=1, col=1)
fig.add_trace(go.Scatter(x=sod2_sub, y=freq2_interp, mode='lines', name=file2_name[0:2]), row=1, col=1)
fig.update_yaxes(title='Freq [Hz]', row=1, col=1)

# panel 2: complex spectrum , with time lag as data and coherence as background
fig.add_trace(go.Contour(x=coh.time, y=coh.scale, z=np.flipud(np.rot90(coh.wtc)), \
    colorscale='magma', colorbar_title_text='Coherence', colorbar_y=0.5, colorbar_len=0.2), \
        row=2, col=1)
fig.add_trace(go.Heatmap(x=coh.time, y=coh.scale, z=time_lag, \
    colorscale='RdBu', colorbar_title_text='Lag [s]', colorbar_y=0.1, colorbar_len=0.2), \
        row=2, col=1)
# plot cone of influence
fig.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash')), row=2, col=1)
fig.update_yaxes(range=scale_range, type='log', title='Scale [s]', row=2, col=1)

# panel 3: time lag spectrum
fig.add_trace(go.Contour(x=coh.time, y=coh.scale, z=time_lag, \
    colorscale='RdBu', colorbar_title_text='Lag [s]', colorbar_y=0.1, colorbar_len=0.2), \
        row=3, col=1)
fig.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash')), row=3, col=1)
fig.update_yaxes(range=scale_range, type='log', title='Scale [s]', row=3, col=1)

# figure layout
fig.update_xaxes(title_text='Time [s]', row=3, col=1)
fig.update_layout(title={'text': file2_name[0:2] + ' relative to ' + file1_name[0:2] + '-Freq [Hz]','x': 0.5, 'y': 0.95})

if save_or_not == 1:
    fig.write_html(save_dir + file1_name[0:2] + '-' + file2_name[0:2] + '-' + str_beg + '-' + str_end + '-Summary.html')

fig.show()