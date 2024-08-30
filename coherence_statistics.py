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
coh_time, scale = df_coh['coh_time'], df_coh['scale']
lag, vel, sign, vel_type  = df_coh['lag'], df_coh['vel'], df_coh['sign'], df_coh['type']

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

# Plotting quasi-radial velocity distribution
plt.figure(figsize=(10, 6))

# Binning velocity every 50km/s, scale every 20s
plt.hist2d(vel_qr, scale_qr, bins=[72, 40], range=[[-1800, 1800], [0, 800]], cmap='jet')
plt.colorbar(label='counts')
plt.xlabel('Vr (km/s)')
plt.ylabel('Scale (s)')
plt.title('Quasi-radial (0-30 deg.) Velocity')

# Plotting inclined and quasi-tangential velocity distribution
plt.figure(figsize=(12, 6))

# Binning velocity every 50km/s, scale every 20s
plt.subplot(1,2,1)
plt.hist2d(np.abs(vel_ic), scale_ic, bins=[36, 40], range=[[0, 1800], [0, 800]], cmap='jet')
plt.colorbar(label='counts')
plt.xlabel('|Vi| (km/s)')
plt.ylabel('Scale (s)')
plt.title('Inclined (30-60 deg.) Velocity')

# Binning velocity every 100km/s, scale every 40s
plt.subplot(1,2,2)
plt.hist2d(np.abs(vel_qt), scale_qt, bins=[18, 20], range=[[0, 1800], [0, 800]], cmap='jet')
cb_qt = plt.colorbar(label='counts')
cb_qt.set_ticks([0,1,2,3])
plt.xlabel('|Vt| (km/s)')
plt.ylabel('Scale (s)')
plt.title('Quasi-tangential (60-90 deg.) Velocity')

plt.show()

# db

