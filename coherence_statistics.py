import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Importing data
file_dir = 'E:/Research/Work/tianwen_IPS/'
file_name = 'wavelet_coherence.xlsx'

data = pd.ExcelFile(file_dir + file_name)
df_dis = pd.read_excel(data, sheet_name='distribution', header=0)
# print('df_dis columns: ', df_dis.columns)
df_coh = pd.read_excel(data, sheet_name='coherency', header=0)
# print('df_coh columns: ', df_coh.columns)

date, stations, length, acute_angle = df_coh['date'], df_coh['stations'], df_coh['length'], df_coh['acute_angle']
int_time = df_coh['integral_time']
coh_time, scale = df_coh['coh_time'], df_coh['scale']
lag, vel, sign, vel_type  = df_coh['lag'], df_coh['vel'], df_coh['sign'], df_coh['type']

# Eliminating fallible data with 'lag < 3 * integral_time'
for i_case in range(len(lag)):
    if np.abs(lag[i_case]) < 3 * int_time[i_case]:
        vel[i_case] = np.nan

# Defining radial-positive velocity
vel_sign = vel * sign

# Dividing acute angle into three bins
bins = [0, 30, 60, 90]
scale, vel_sign = np.array(scale), np.array(vel_sign)
# Quasi-radial bin
ind_qr = np.where((acute_angle > bins[0]) & (acute_angle < bins[1]))
scale_qr, vel_qr = scale[ind_qr], vel_sign[ind_qr]
# Inclined bin
ind_ic = np.where((acute_angle > bins[1]) & (acute_angle < bins[2]))
scale_ic, vel_ic = scale[ind_ic], vel_sign[ind_ic]
# Quasi-tangential bin
ind_qt = np.where((acute_angle > bins[2]) & (acute_angle < bins[3]))
scale_qt, vel_qt = scale[ind_qt], vel_sign[ind_qt]

# Counting inward and outward propagation
num_outward = np.sum(np.array(vel_qr) > 0)
num_inward = np.sum(np.array(vel_qr) < 0)
outward_perc = num_outward / (num_outward + num_inward)# 87.9%

# Plotting quasi-radial velocity distribution
plt.figure(figsize=(10, 4))

# Binning velocity every 50km/s, scale every 20s
plt.hist2d(vel_qr, scale_qr, bins=[72, 40], range=[[-1800, 1800], [0, 800]], cmap='jet')
plt.colorbar(label='counts')
plt.axvline(x=0, ymin=0, ymax=800, color='white', linewidth=2)
plt.xlim([-1000, 1000])
plt.xlabel('$v_{proj}$ (km/s)')
plt.ylabel('Scale (s)')
plt.title('Along quasi-radial baselines')

# Plotting inclined and quasi-tangential velocity distribution
plt.figure(figsize=(10, 4))

# Binning velocity every 50km/s, scale every 20s
plt.subplot(1,2,1)
plt.hist2d(np.abs(vel_ic), scale_ic, bins=[36, 40], range=[[0, 1800], [0, 800]], cmap='jet')
plt.colorbar(label='counts')
plt.xlim([0, 1500])
plt.xlabel('|$v_{proj}$| (km/s)')
plt.ylabel('Scale (s)')
plt.title('Along oblique baselines')

# Binning velocity every 100km/s, scale every 40s
plt.subplot(1,2,2)
plt.hist2d(np.abs(vel_qt), scale_qt, bins=[18, 20], range=[[0, 1800], [0, 800]], cmap='jet')
cb_qt = plt.colorbar(label='counts')
cb_qt.set_ticks([0,1,2,3])
plt.xlim([0, 1500])
plt.xlabel('|$v_{proj}$| (km/s)')
plt.ylabel('Scale (s)')
plt.title('Along quasi-tangential baselines')

plt.show()

db

