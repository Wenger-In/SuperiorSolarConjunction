import scipy.io
import numpy as np

date = 's1a26x'
station = 'ur'

read_dir = 'E:/Research/Data/Tianwen/' + date +'/' + station + '/chan3/'
read_name = 'Time_Frequency_Phase.mat'
save_dir = 'E:/Research/Data/Tianwen/' + date + '/'
save_name = station + station + 'chan3_1frephase.dat'

mat_data = scipy.io.loadmat(read_dir + read_name)
data_to_save = mat_data['Fre_Pha_data']
np.savetxt(save_dir + save_name, data_to_save, fmt='%f')

data = np.loadtxt(save_dir + save_name)