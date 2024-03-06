import matplotlib.pyplot as plt
import pandas as pd
import pyleoclim as pyleo
import numpy as np
from matplotlib.colors import SymLogNorm
from scipy.stats import zscore

## function: convert the time data to 'seconds of day'
def convert_to_second_of_day(time_array):
    if isinstance(time_array, int):
        time_array = [time_array]
    sod_array = []
    for time in time_array:
        time = int(time)
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
i_case = 2
if i_case == 1: # 2021/09/30(273), 12:00-13:00, Ht-Sv, Ht-Wz, Sv-Wz, (latitudinal fluctuaion)
    file_dir = 'E:/Research/Data/Tianwen/m1930x_up_new/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1930x_up_new/'
    file_Ht = 'HtHtchan3_1frephase1s.dat'
    file_Sv = 'SvSvchan3_1frephase1s.dat'
    file_Wz = 'WzWzchan3_1frephase1s.dat'
    file1_name = file_Ht
    file2_name = file_Sv
    sub_beg = convert_to_second_of_day(2021273120000)[0]
    sub_end = convert_to_second_of_day(2021273130000)[0]
elif i_case == 2: # 2021/10/04(277), 05:40-08:20, Js-Bd, Bd-Yg, Yg-Hh, (inward propagation, latitudinal fluctuaion, outward propagation)
    file_dir = 'E:/Research/Data/Tianwen/m1a04x_renew/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a04x_renew/'
    file_Js = 'JsJschan3_1frephase1s.dat' # time has been formatted as 'seconds of day'
    file_Bd = 'BdBdchan3_1frephase1s.dat'
    file_Yg = 'YgYgchan3_1frephase2s.dat'
    file_Hh = 'HhHhchan3_1frephase1s.dat'
    file1_name = file_Yg
    file2_name = file_Hh
    sub_beg = convert_to_second_of_day(2021277073000)[0]
    sub_end = convert_to_second_of_day(2021277080000)[0]
elif i_case == 3: # 2021/10/07(280), 03:30-04:00, sh-km, (polar region fluctuation)
    file_dir = 'E:/Research/Data/Tianwen/m1a07x_up/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a07x_up/'
    file_sh = 'shshchan3_1frephase5s.dat'
    file_km = 'kmkmchan3_1frephase5s.dat'
    file1_name = file_sh
    file2_name = file_km
    sub_beg = convert_to_second_of_day(2021280033000)[0]
    sub_end = convert_to_second_of_day(2021280040000)[0]
elif i_case == 4: # 2021/10/15(288), 01:00-04:00, Bd-Hh, Bd-Ys, Hh-Ys, (fine structure)
                  #                  07:40-13:00, Bd-Hh, Bd-Ys, Hh-Ys, (CME)
    file_dir = 'E:/Research/Data/Tianwen/m1a15y_copy/'
    save_dir = 'E:/Research/Work/tianwen_IPS/m1a15y_copy/'
    file_Bd = 'BdBdchan3_1frephase1s.dat'
    file_Hh = 'HhHhchan3_1frephase1s.dat'
    file_Ys = 'YsYschan3_1frephase1s.dat'
    file1_name = file_Bd
    file2_name = file_Hh
    sub_beg = convert_to_second_of_day(2021288010000)[0]
    sub_end = convert_to_second_of_day(2021288013000)[0]

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
freq1_sub, sod1_sub = eliminate_outliers(freq1_sub, sod1_sub, 2)
freq2_sub, sod2_sub = eliminate_outliers(freq2_sub, sod2_sub, 2)

## detrend for frequency sequence
freq1_sub = detrend(freq1_sub, sod1_sub, 2, file1_name + '-Freq[Hz]')
freq2_sub = detrend(freq2_sub, sod2_sub, 2, file2_name + '-Freq[Hz]')

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

## figure 1: summary plot of series1
fig1, ax1 = series1.summary_plot(figsize=[10,5], psd=psd1, scalogram=cwt1, title=file1_name[0:2])

## figure 2: summary plot of series2
fig2, ax2 = series2.summary_plot(figsize=[10,5], psd=psd2, scalogram=cwt2, title=file2_name[0:2])

## figure 3: coherence spectrum between series1 and series2
coh.plot(figsize=[12,3], signif_thresh=0.9)

## figure 4: significance test of coherence spectrum
# cwt_sig = coh.signif_test(number=3, qs=[.9,.95])
# cwt_sig.plot(figsize=[12,3], signif_thresh = 0.9)

## figure 5: phase and time lag spectrum
plt.figure(figsize=[15,6])
# phase spectrum
plt.subplot(2,1,1)
plt.contour(coh.time, coh.scale, np.flipud(np.rot90(coh.wtc)), levels=3, cmap='YlGn') # coherence contours overlap in graph
cb11 = plt.colorbar()
cb11.set_label('WTC')
plt.pcolor(coh.time, coh.scale, np.flipud(np.rot90(np.degrees(coh.phase))), cmap='RdBu') # phase pcolors as the background
cb12 = plt.colorbar()
cb12.set_label('Phase [rad.]')
plt.plot(coh.time, coh.coi, 'k--') # cone of influence
plt.xlabel('time [s]')
plt.ylabel('scale [s]')
plt.ylim([np.min(coh.scale), np.max(coh.scale)])
plt.yscale('log')
# time lag spectrum
plt.subplot(2,1,2)
plt.contour(coh.time, coh.scale, np.flipud(np.rot90(coh.wtc)), levels=3,
            cmap='YlGn', norm=SymLogNorm(linthresh=1)) # coherence contours overlap in graph
cb21 = plt.colorbar()
cb21.set_label('WTC')
plt.pcolor(coh.time, coh.scale, np.flipud(np.rot90(coh.phase/2/np.pi/coh.frequency[np.newaxis,:])), 
               cmap='RdBu', norm=SymLogNorm(linthresh=1))  # time lag pcolors as the background
cb22 = plt.colorbar()
cb22.set_label('lag [s]')
plt.plot(coh.time, coh.coi, 'k--') # cone of influence
plt.xlabel('time [s]')
plt.ylabel('scale [s]')
plt.ylim([np.min(coh.scale), np.max(coh.scale)])
plt.yscale('log')

plt.suptitle(file1_name[0:2] + '-' + file2_name[0:2] + '-Freq [Hz]')

plt.show()

db