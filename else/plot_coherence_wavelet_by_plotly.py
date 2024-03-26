import matplotlib.pyplot as plt
import pandas as pd
import pyleoclim as pyleo
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def linear_detrend(phase_sequence, time, title):
    coefficients = np.polyfit(time, phase_sequence, 1)
    fit_sequence = np.polyval(coefficients, time)
    detrended_sequence = phase_sequence - fit_sequence
    plt.figure()
    plt.plot(time,phase_sequence,label='phase')
    plt.plot(time,fit_sequence,label='fit')
    plt.legend()
    plt.title(title)
    # plt.show()
    plt.close()
    return detrended_sequence

def quadratic_detrend(frequency_sequence,time,title):
    frequency_mean = np.mean(frequency_sequence)
    frequency_sequence[np.abs(frequency_sequence-frequency_mean)>2] = 0
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
    # plt.show()
    plt.close()
    return detrended_sequence

# import data
file_dir = 'E:/Research/Data/Tianwen/m1a04x_renew/'
file1_name = 'BdBdchan3_1frephase1s.dat'
file2_name = 'HtHtchan3_1frephase1s.dat'
var_list = ['Residual Frequency', 'Residual Phase', 'Signal Density', 'Noise Density']
unit_list = ['Hz', 'rad', 'dB', 'dB']

save_dir = 'E:/Research/Work/tianwen_IPS/m1a04x_renew/'

# read data
file1_path = file_dir + file1_name
file2_path = file_dir + file2_name
data1 = np.loadtxt(file1_path)
data2 = np.loadtxt(file2_path)
t = np.linspace(0, len(data1), len(data1))
slt_indices = np.arange(500,3500) # select time interval of steady sequence
t = t[slt_indices]
var1 = data1[slt_indices, 1:]
var2 = data2[slt_indices, 1:]
freq1 = var1[:,0]
freq2 = var2[:,0]

# variable name
var_name = var_list[0]
unit_name =  unit_list[0]
title = var_name

# quadratic detrend for frequency sequence
freq1 = quadratic_detrend(freq1, t, title)
freq2 = quadratic_detrend(freq2, t, title)

# construct pyleo.series         
series1 = pyleo.Series(time=t,value=freq1,time_name='t',
					time_unit='second',value_name = 'freq', 
					value_unit='Hz',label='Bd')
series2 = pyleo.Series(time=t,value=freq2,time_name='t',
					time_unit='second',value_name = 'freq', 
					value_unit='Hz',label='Hh')

# continuous wavelet translation
cwt1 = series1.wavelet()
cwt2 = series2.wavelet()

# wavelet coherence
coh = series2.wavelet_coherence(series1,method='cwt')

# plotly figures
fig1 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

fig1.add_trace(go.Scatter(x=t, y=freq1, mode='lines', name='Bd'), row=1, col=1)
fig1.update_yaxes(title='Freq [Hz]', row=1, col=1)
fig1.add_trace(go.Scatter(x=t, y=freq2, mode='lines', name='Hh'), row=2, col=1)
fig1.update_yaxes(title='Freq [Hz]', row=2, col=1)

fig1.add_trace(go.Contour(x=cwt1.time, y=cwt1.scale, z=np.flipud(np.rot90(cwt1.amplitude)), 
                         colorscale='jet', colorbar_title_text='CWT_1', colorbar_y=0.4, colorbar_len=0.3),
              row=3, col=1)
fig1.update_yaxes(type='log', title='Scale [s]', row=3, col=1)
fig1.add_trace(go.Contour(x=cwt2.time, y=cwt2.scale, z=np.flipud(np.rot90(cwt2.amplitude)),
                         colorscale='jet', colorbar_title_text='CWT_2', colorbar_y=0.1, colorbar_len=0.3),
              row=4, col=1)
fig1.update_yaxes(type='log', title='Scale [s]', row=4, col=1)
fig1.update_xaxes(title_text='Time [s]', row=4, col=1)

fig1.show()
# fig1.write_html(save_dir+'cwt.html')

# plotly figures
fig2 = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02)

fig2.add_trace(go.Scatter(x=t, y=freq1, mode='lines', name='Bd'), row=1, col=1)
fig2.update_yaxes(title='Freq [Hz]', row=1, col=1)
fig2.add_trace(go.Scatter(x=t, y=freq2, mode='lines', name='Hh'), row=2, col=1)
fig2.update_yaxes(title='Freq [Hz]', row=2, col=1)

fig2.add_trace(go.Contour(x=coh.time, y=coh.scale, z=np.flipud(np.rot90(coh.wtc)), 
                         colorscale='jet', colorbar_title_text='Coherence', colorbar_y=0.5, colorbar_len=0.2),
              row=3, col=1)
fig2.update_yaxes(type='log', title='Scale [s]', row=3, col=1)
fig2.add_trace(go.Contour(x=coh.time, y=coh.scale, z=np.flipud(np.rot90(coh.phase)),
                         colorscale='RdBu', colorbar_title_text='Phase [rad.]', colorbar_y=0.3, colorbar_len=0.2), 
              row=4, col=1)
fig2.update_yaxes(type='log', title='Scale [s]', row=4, col=1)
fig2.add_trace(go.Contour(x=coh.time, y=coh.scale, z=np.flipud(np.rot90(coh.phase/2/np.pi/coh.frequency[np.newaxis,:])), 
                         colorscale='RdBu', colorbar_title_text='Lag [s]', colorbar_y=0.1, colorbar_len=0.2), 
              row=5, col=1)
fig2.update_yaxes(type='log', title='Scale [s]', row=5, col=1)
fig2.update_xaxes(title_text='Time [s]', row=5, col=1)

fig2.show()
# fig2.write_html(save_dir+'wtc.html')