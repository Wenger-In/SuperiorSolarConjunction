import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Importing data
file_dir = 'E:/Research/Work/tianwen_IPS/'
file_name = 'multiple_base_line.xlsx'
Rs_km = 696000

data = pd.ExcelFile(file_dir + file_name)
df_coh = pd.read_excel(data, sheet_name='coherency', header=0)
# print('df_coh columns: ', df_coh.columns)
df_info = pd.read_excel(data, sheet_name='station_info', header=0)
# print('df_info columns: ', df_info.columns)

vel = np.array(df_coh['vel'])
proj_x = np.array(df_info['proj_x'])
proj_y = np.array(df_info['proj_y'])

def unit_vector(point_str, point_end):
    vector = point_end - point_str
    magnitude = np.linalg.norm(vector)
    unit_vector = vector / magnitude
    return unit_vector

def plot_projection(proj_x, proj_y, colors):
    plt.plot([0, proj_x[0] / Rs_km], [0, proj_y[0] / Rs_km], linewidth=2, c=colors[0])
    plt.plot([0, proj_x[1] / Rs_km], [0, proj_y[1] / Rs_km], linewidth=2, c=colors[1])
    plt.plot([0, proj_x[2] / Rs_km], [0, proj_y[2] / Rs_km], linewidth=2, c=colors[2])
    plt.plot([proj_x[0] / Rs_km, proj_x[1] / Rs_km], [proj_y[0]/ Rs_km, proj_y[1] / Rs_km], 'k--', linewidth=2)
    plt.plot([proj_x[0] / Rs_km, proj_x[2] / Rs_km], [proj_y[0]/ Rs_km, proj_y[2] / Rs_km], 'k--', linewidth=2)
    plt.plot([proj_x[1] / Rs_km, proj_x[2] / Rs_km], [proj_y[1]/ Rs_km, proj_y[2] / Rs_km], 'k--', linewidth=2)
    
def plot_vel(point_str, vel, color, scale=100000):
    plt.quiver(*point_str, *vel, angles='xy', scale_units='xy', scale=scale, color=color, width=0.005)

def select_start_point(proj_x):
    str_idx = np.argmax(np.abs(proj_x))
    return str_idx

def correct_vel_direction(phi, vel):
    phi_corr, vel_corr = phi, vel
    for i_line in range(len(phi)):
        if vel[i_line] < 0:
            phi_corr[i_line] = phi[i_line] + np.pi
            vel_corr[i_line] = -vel[i_line]
    return phi_corr, vel_corr

def LSM_optimization(e01, e02, e12, vel):
    # Calculating base line angle
    phi01 = np.arctan2(e01[1], e01[0])
    phi02 = np.arctan2(e02[1], e02[0])
    phi12 = np.arctan2(e12[1], e12[0])
    phi = np.array([phi01, phi02, phi12])
    # Correcting the direction of velocity (to ensure positive velocity)
    phi_corr, vel_corr = correct_vel_direction(phi, vel)
    # Giving optimization range
    vp_range = np.linspace(10, 500, 98)
    theta_range = np.linspace(0, 2*np.pi, 360)
    # Calculating loss function
    S_loss = np.zeros((len(vp_range), len(theta_range)))
    for i, vp in enumerate(vp_range):
        for j, theta in enumerate(theta_range):
            cos_term = np.cos(theta - phi_corr)
            # Avioding dividing by zero
            cos_term = np.clip(cos_term, 1e-10, None)
            residual = vel_corr - vp / cos_term
            S_loss[i, j] = np.sum(residual**2)
    return S_loss, vp_range, theta_range

def plot_loss(S_loss, vp_range, theta_range, i_case):
    theta_deg_range = np.degrees(theta_range) # Converting to degrees
    min_idx = np.unravel_index(np.argmin(S_loss), S_loss.shape)
    min_vp = vp_range[min_idx[0]]
    min_theta_deg = theta_deg_range[min_idx[1]]
    
    Vp, Theta = np.meshgrid(vp_range, theta_deg_range)
    min_loss = np.min(S_loss)
    S_loss_norm = S_loss / min_loss
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(Vp, Theta, S_loss_norm.T, shading='auto', cmap='jet', norm=LogNorm())
    plt.colorbar(label='Loss')
    
    plt.scatter(min_vp, min_theta_deg, color='k', marker='x', s=50, \
        label=f'Min Value: $S_{{min}}={min_loss:.2e}$ at $v_p={min_vp:.2f}$km/s, $\\theta={min_theta_deg:.2f}^\circ$ @ case'+str(i_case+1))
    plt.legend()
    
    plt.xlabel('Phase Speed $v_p$ (km/s)')
    plt.ylabel('Propagation Angle $\\theta$ (deg.)')
    plt.title('Loss Function $S(v_p, \\theta)$')
    return min_loss, min_vp, min_theta_deg

def plot_perpendicular_line(point_str, vel_opt, scale, line_length=2):
    endpoint = point_str + vel_opt / scale
    perp_vector = np.array([-vel_opt[1], vel_opt[0]])
    perp_vector = perp_vector / np.linalg.norm(perp_vector) * line_length
    
    line_start = endpoint - perp_vector / 2
    line_end = endpoint + perp_vector / 2
    
    plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'm--')

for i_case in range(2,3):
    vel_case = vel[i_case*4:i_case*4+3]
    proj_x_case = proj_x[i_case*4:i_case*4+3]
    proj_y_case = proj_y[i_case*4:i_case*4+3]
    
    e01 = unit_vector(np.array([proj_x_case[0],proj_y_case[0]]), np.array([proj_x_case[1],proj_y_case[1]]))
    e02 = unit_vector(np.array([proj_x_case[0],proj_y_case[0]]), np.array([proj_x_case[2],proj_y_case[2]]))
    e12 = unit_vector(np.array([proj_x_case[1],proj_y_case[1]]), np.array([proj_x_case[2],proj_y_case[2]]))
    
    # Consist to the positive direction of unit vectors!
    vel_case = -vel_case
    vel01 = vel_case[0] * e01
    vel02 = vel_case[1] * e02
    vel12 = vel_case[2] * e12

    S_loss, vp_range, theta_range = LSM_optimization(e01, e02, e12, vel_case)
    min_loss, min_vp, min_theta_deg = plot_loss(S_loss, vp_range, theta_range, i_case)
    vel_opt_x, vel_opt_y = min_vp * np.cos(np.radians(min_theta_deg)), min_vp * np.sin(np.radians(min_theta_deg))
    vel_opt = np.array([vel_opt_x, vel_opt_y])
    
    plt.figure(figsize=(6,6))
    
    plot_projection(proj_x_case, proj_y_case, 'rgb')
    
    str_idx = select_start_point(proj_x_case)
    point_str = np.array([proj_x_case[str_idx] / Rs_km, proj_y_case[str_idx] / Rs_km])
    
    scale = 100000
    plot_vel(point_str, vel01, 'orange', scale)
    plot_vel(point_str, vel02, 'orange', scale)
    plot_vel(point_str, vel12, 'orange', scale)
    plot_vel(point_str, vel_opt, 'purple', scale)
    plot_perpendicular_line(point_str, vel_opt, scale)
    
    plt.xlim(proj_x_case[str_idx] / Rs_km - 0.01, proj_x_case[str_idx] / Rs_km + 0.01)
    plt.ylim(proj_y_case[str_idx] / Rs_km - 0.01, proj_y_case[str_idx] / Rs_km + 0.01)
    
    plt.gca().set_aspect(1)
    plt.xlabel('X (Rs)')
    plt.ylabel('Y (Rs)')
    plt.title('Plane of Sky @ case' + str(i_case+1))
    
    plt.show()

db