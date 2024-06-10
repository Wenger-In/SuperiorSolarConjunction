import matplotlib.pyplot as plt
import pandas as pd
import pyleoclim as pyleo
import numpy as np
from matplotlib.colors import SymLogNorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import statistics

sys.path.append(r'E:/Research/Program/SuperiorSolarConjunction')
import frequency_analyse_utils
from frequency_analyse_utils import convert_to_second_of_day, convert_to_HHMM, \
    eliminate_outliers, interpolate, detrend

## import data
i_case = 3
save_or_not = 1
if i_case == 1: # 2021/09/30(273), 12:00-13:00, Ht-Sv, Ht-Wz, Sv-Wz, (latitudinal fluctuaion)
    file_dir = 'E:/Research/Data/Tianwen/m1930x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1930x/'
    file_Ht = 'HtHtchan3_1frephase1s.dat'
    file_Sv = 'SvSvchan3_1frephase1s.dat'
    file_Wz = 'WzWzchan3_1frephase1s.dat'
    file1_name = file_Sv
    file2_name = file_Wz
    time_beg = 2021273123000
    time_end = 2021273130000
elif i_case == 2: # 2021/10/03(276), 09:30-10:00, Ht, Wz, Zc, (latitudinal fluctuaion)
    file_dir = 'E:/Research/Data/Tianwen/m1a03x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a03x/'
    file_Ht = 'HtHtchan3_1frephase1s.dat'
    file_Wz = 'WzWzchan3_1frephase1s.dat'
    file_Zc = 'ZcZcchan3_1frephase1s.dat'
    file1_name = file_Wz
    file2_name = file_Zc
    time_beg = 2021273093000
    time_end = 2021273100000
elif i_case == 3: # 2021/10/04(277), 05:40-08:20, Js-Bd, Bd-Yg, Yg-Hh, (inward propagation, latitudinal fluctuaion, outward propagation)
    file_dir = 'E:/Research/Data/Tianwen/m1a04x_renew/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a04x_renew/'
    file_Js = 'JsJschan3_1frephase1s.dat' # time has been formatted as 'seconds of day'
    file_Bd = 'BdBdchan3_1frephase1s.dat'
    file_Yg = 'YgYgchan3_1frephase4s.dat'
    file_Hh = 'HhHhchan3_1frephase4s.dat' # 'HhHhchan3_1frephase1s.dat'
    file1_name = file_Yg
    file2_name = file_Hh
    time_beg = 2021277064100
    time_end = 2021277065500
elif i_case == 4: # 2021/10/07(280), 03:30-04:00, sh-km, (polar region fluctuation)
    file_dir = 'E:/Research/Data/Tianwen/m1a07x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a07x/'
    file_sh = 'shshchan3_1frephase5s.dat'
    file_km = 'kmkmchan3_1frephase5s.dat'
    file1_name = file_sh
    file2_name = file_km
    time_beg = 2021280041500
    time_end = 2021280044500
elif i_case == 5: # 2021/10/15(288), 01:00-04:00, hb, sh, km, (fine structure)
    file_dir = 'E:/Research/Data/Tianwen/m1a15x/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a15x/'
    file_hb = 'hbhbchan3_1frephase5s.dat'
    file_ke = 'kekesignum111frephase5s.dat'
    file_sh = 'shshchan3_1frephase1s.dat'
    file_km = 'kmkmchan3_1frephase1s.dat'
    file1_name = file_sh
    file2_name = file_km
    time_beg = 2021288024500
    time_end = 2021288031500
elif i_case == 6: # 2021/10/15(288), 07:40-13:00, Bd-Ys, Bd-Hh, Ys-Hh, Js-Bd, Js-Ys, Js-Hh, (CME)
    file_dir = 'E:/Research/Data/Tianwen/m1a15y/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a15y/'
    file_Bd = 'BdBdchan3_1frephase1s.dat'
    file_Ys = 'YsYschan3_1frephase1s.dat'
    file_Hh = 'HhHhchan3_1frephase1s.dat'
    file_Js = 'JsJschan3_1frephase1s.dat'
    file1_name = file_Ys
    file2_name = file_Hh
    time_beg = 2021288110000
    time_end = 2021288112200
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
if i_case == 6 and file2_name == 'HhHhchan3_1frephase1s.dat':
    sod2 = time2 # time has been formatted as 'seconds of day'
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
freq1_out, sod1_out = eliminate_outliers(freq1_sub, sod1_sub, 10) #3
freq2_out, sod2_out = eliminate_outliers(freq2_sub, sod2_sub, 10) #3
# step 2: detrend for frequency sequence
freq1_fit, freq1_detrend = detrend(freq1_out, sod1_out, 5)
freq1_fit, freq2_detrend = detrend(freq2_out, sod2_out, 5)
# step 3: interpolation for frequency sequence
freq1_interp = interpolate(freq1_detrend, sod1_out, sod1_sub)
freq2_interp = interpolate(freq2_detrend, sod2_out, sod2_sub)

# ## calculate standard deviation
std_dev1 = statistics.stdev(freq1_interp)
std_dev2 = statistics.stdev(freq2_interp)
print("std_dev_1=", std_dev1)
print("std_dev_2=", std_dev2)

## construct pyleo.series         
series1 = pyleo.Series(time=sod1_sub, value=freq1_interp, \
    time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file1_name[0:2])
series2 = pyleo.Series(time=sod2_sub, value=freq2_interp, \
    time_name='Time', time_unit='s', value_name = 'Freq', value_unit='Hz', label=file2_name[0:2])

# ## transform analysis
# # Fourier transform
# fft1 = series1.spectral(method='cwt')
# fft2 = series2.spectral(method='cwt')
# # wavelet transform
# cwt1 = series1.wavelet(method='cwt')
# cwt2 = series2.wavelet(method='cwt')

## wavelet coherence analysis
coh = series2.wavelet_coherence(series1, method='cwt')
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
    colorscale='RdBu', zmin=-np.max(np.abs(time_lag)), zmax=np.max(np.abs(time_lag)), \
        colorbar_title_text='Lag [s]', colorbar_y=0.1, colorbar_len=0.3), \
            row=2, col=1)
# plot cone of influence
fig.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash'), name='coi'), \
    row=2, col=1)
fig.update_yaxes(range=scale_range, type='log', title='Scale [s]', row=2, col=1)

# panel 3: time lag spectrum
fig.add_trace(go.Contour(x=coh.time, y=coh.scale, z=time_lag, \
    colorscale='RdBu', zmin=-np.max(np.abs(time_lag)), zmax=np.max(np.abs(time_lag)), \
        colorbar_title_text='Lag [s]', colorbar_y=0.1, colorbar_len=0.3), \
            row=3, col=1)
# plot cone of influence
fig.add_trace(go.Scatter(x=coh.time, y=coh.coi, mode='lines', line=dict(dash='dash'), name='coi'), \
    row=3, col=1)
fig.update_yaxes(range=scale_range, type='log', title='Scale [s]', row=3, col=1)

# figure layout
xposs = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
xticks = [int(xpos*(sod_end-sod_beg)+sod_beg) for xpos in xposs]
xlabels = [convert_to_HHMM(xtick) for xtick in xticks]
fig.update_xaxes(title_text='Time [HHMM]', tickvals=xticks, ticktext=xlabels, row=3, col=1)
fig.update_layout(title={'text': file2_name[0:2] + ' relative to ' + file1_name[0:2] + ' on ' + file_dir[25:-1], \
    'x': 0.5, 'y': 0.95})

if save_or_not == 1:
    fig.write_html(save_dir + file1_name[0:2] + '-' + file2_name[0:2] + '-' + str_beg + '-' + str_end + '-4s-Summary.html')
    # fig.write_image(save_dir + file1_name[0:2] + '-' + file2_name[0:2] + '-' + str_beg + '-' + str_end + '-Summary.png')
else:
    fig.show()

# db