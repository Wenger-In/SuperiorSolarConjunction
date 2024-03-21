import matplotlib.pyplot as plt
import pandas as pd
import pyleoclim as pyleo
import numpy as np
from matplotlib.colors import SymLogNorm
from scipy.stats import zscore
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
save_or_not = 1
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

## construct pyleo.series         
series1 = pyleo.Series(time=sod1_sub, value=freq1_sub, time_name='t',
					time_unit='s', value_name = 'Freq', 
					value_unit='Hz', label=file1_name[0:2])
series2 = pyleo.Series(time=sod2_sub, value=freq2_sub, time_name='t',
					time_unit='s', value_name = 'Freq', 
					value_unit='Hz', label=file2_name[0:2])

## power spectral transform
# Fourier transform
psd1 = series1.spectral(method='wwz')
psd2 = series2.spectral(method='wwz')
# wavelet transform
cwt1 = series1.wavelet(method='wwz')
cwt2 = series2.wavelet(method='wwz')

## wavelet coherence
coh = series2.wavelet_coherence(series1, method='wwz')
coh.wtc[coh.wtc>1] = np.nan

# ## figure 1: plots of series1 and series2 together
# fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

# fig1.add_trace(go.Scatter(x=sod1_sub, y=freq1_sub, mode='lines', name='Bd'), row=1, col=1)
# fig1.update_yaxes(title=file1_name[0:2]+' Freq [Hz]', row=1, col=1)
# fig1.add_trace(go.Scatter(x=sod2_sub, y=freq2_sub, mode='lines', name='Hh'), row=2, col=1)
# fig1.update_yaxes(title=file2_name[0:2]+' Freq [Hz]', row=2, col=1)

# fig1.show()

# if save_or_not == 1:
#     fig1.write_html(save_dir + file1_name[0:2] + '-' + file2_name[0:2] + '-' + str_beg + '-' + str_end + '-TimeSeries.html')

## figure 2: coherence and time lag spectrum
fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.02)

# time series
fig2.add_trace(go.Scatter(x=sod1_sub, y=freq1_sub, mode='lines', name=file1_name[0:2]), row=1, col=1)
fig2.update_yaxes(title='Freq [Hz]', row=1, col=1)
fig2.add_trace(go.Scatter(x=sod2_sub, y=freq2_sub, mode='lines', name=file2_name[0:2]), row=1, col=1)
fig2.update_yaxes(title='Freq [Hz]', row=1, col=1)

# # coherence spectrum
# fig2.add_trace(go.Contour(x=coh.time, y=coh.scale, z=np.flipud(np.rot90(coh.wtc)),
#                          colorscale='magma', colorbar_title_text='Coherence', colorbar_y=0.7, colorbar_len=0.2), 
#               row=1, col=1)
# fig2.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash')), row=1, col=1)
# fig2.update_yaxes(range=[np.log10(np.min(coh.scale)), np.log10(np.max(coh.scale))], type='log', title='Scale [s]', row=1, col=1)

# time lag data and coherence background
fig2.add_trace(go.Contour(x=coh.time, y=coh.scale, z=np.flipud(np.rot90(coh.wtc)),
                         colorscale='magma', colorbar_title_text='Coherence', colorbar_y=0.5, colorbar_len=0.2), 
              row=2, col=1)
fig2.add_trace(go.Heatmap(x=coh.time, y=coh.scale, z=np.flipud(np.rot90(coh.phase/2/np.pi/coh.frequency[np.newaxis,:])), 
                         colorscale='RdBu', colorbar_title_text='Lag [s]', colorbar_y=0.1, colorbar_len=0.2), 
              row=2, col=1)
fig2.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash')), row=2, col=1)
fig2.update_yaxes(range=[np.log10(np.min(coh.scale)), np.log10(np.max(coh.scale))], type='log', title='Scale [s]', row=2, col=1)

# time lag spectrum
fig2.add_trace(go.Contour(x=coh.time, y=coh.scale, z=np.flipud(np.rot90(coh.phase/2/np.pi/coh.frequency[np.newaxis,:])), 
                         colorscale='RdBu', colorbar_title_text='Lag [s]', colorbar_y=0.1, colorbar_len=0.2), 
              row=3, col=1)
fig2.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash')), row=3, col=1)
fig2.update_yaxes(range=[np.log10(np.min(coh.scale)), np.log10(np.max(coh.scale))], type='log', title='Scale [s]', row=3, col=1)

fig2.update_xaxes(title_text='Time [s]', row=3, col=1)
fig2.update_layout(title={'text': file2_name[0:2] + ' relative to ' + file1_name[0:2] + '-Freq [Hz]','x': 0.5, 'y': 0.95})

if save_or_not == 1:
    # fig2.write_html(save_dir + file1_name[0:2] + '-' + file2_name[0:2] + '-' + str_beg + '-' + str_end + '-PhaseLag.html')
    fig2.write_html(save_dir + file1_name[0:2] + '-' + file2_name[0:2] + '-' + str_beg + '-' + str_end + '-Summary.html')

# fig2.show()

# db