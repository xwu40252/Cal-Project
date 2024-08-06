# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:55:42 2024

@author: Nierrr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def pts(arrival_profile, traffic_signal_state):
    a = [0] + arrival_profile
    s = [0] + traffic_signal_state
    T = len(a)
    max_queue = 20

    x_t = np.zeros((T, max_queue))
    b_t = np.zeros(T)
    x_t[0][0] = 1.0

    for t in range(1, T):
        new_x_t = np.zeros(max_queue)
        for k in range(max_queue):
            if k > 0:
                new_x_t[k] += (x_t[t-1][k-1] * a[t] + x_t[t-1][k] * (1 - a[t]))
        
        b_t[t] = np.sum(new_x_t[1:] * s[t])
        if s[t] == 1:
            for k in range(1, max_queue):
                new_x_t[k-1] += new_x_t[k]
                new_x_t[k] = 0
        
        x_t[t] = new_x_t

    return x_t.tolist(), b_t.tolist()

def calculate_traffic_signal_state(cycle_length, green_start, green_end):
    traffic_signal_state = [0] * cycle_length
    for i in range(green_start - 1, green_end):
        if i < cycle_length:
            traffic_signal_state[i] = 1
    return traffic_signal_state

def calculate_arrival_profile(data, cycle_length, delta_u, phi):
    data = data[['Simulation Time', 'Vehicle Number', 'Position', 'Speed']]
    N_c = int(data['Simulation Time'].max() // cycle_length) + 1
    unique_vehicle_numbers = data['Vehicle Number'].unique()
    selected_vehicles = np.random.choice(unique_vehicle_numbers, size=int(phi * len(unique_vehicle_numbers)), replace=False)
    sampled_data = data[data['Vehicle Number'].isin(selected_vehicles)]
    sampled_data = sampled_data.sort_values(by=['Vehicle Number', 'Simulation Time']).drop_duplicates(subset=['Vehicle Number'], keep='first')
    sampled_data['TimeInCycle'] = sampled_data['Simulation Time'] % cycle_length
    arrivals_per_time_point = sampled_data.groupby('TimeInCycle')['Vehicle Number'].nunique().reindex(np.arange(cycle_length), fill_value=0)
    arrival_profile = (arrivals_per_time_point / (N_c * delta_u * phi)).tolist()
    return arrival_profile

def calculate_queue_length_distribution(data, cycle_length, delta_u):
    data['TimeInCycle'] = data['Simulation Time'] % cycle_length
    data['Cycle'] = data['Simulation Time'] // cycle_length
    queue_lengths = defaultdict(list)
    
    for cycle in range(int(data['Cycle'].max()) + 1):
        cycle_data = data[data['Cycle'] == cycle]
        for time_in_cycle in range(cycle_length):
            time_data = cycle_data[cycle_data['TimeInCycle'] == time_in_cycle]
            queue_length = np.sum(time_data['Speed'] <= 5) * delta_u
            queue_lengths[time_in_cycle].append(queue_length)
    
    queue_length_distribution = {}
    max_length = max(max(lengths) for lengths in queue_lengths.values())
    for time_point, lengths in queue_lengths.items():
        length_counts = pd.Series(lengths).value_counts(normalize=True).reindex(np.arange(max_length+1), fill_value=0)
        queue_length_distribution[time_point] = length_counts.tolist()
    
    return queue_length_distribution

def calculate_error(est_queue_data, sim_queue_data):
    est_queue_data_np = np.array(est_queue_data)
    sim_queue_data_np = np.array(sim_queue_data)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_error = np.abs(((est_queue_data_np - sim_queue_data_np[:, :est_queue_data_np.shape[1]]) / sim_queue_data_np[:, :est_queue_data_np.shape[1]])) *100
        percentage_error = np.nan_to_num(percentage_error, nan=0.0, posinf=0.0, neginf=0.0)
    return percentage_error

def process_data(file_path, cycle_length, green_start, green_end, delta_u, phi):
    data = pd.read_csv(file_path)
    
    arrival_profile = calculate_arrival_profile(data, cycle_length, delta_u, phi)
    traffic_signal_state = calculate_traffic_signal_state(cycle_length, green_start, green_end)
    
    x_t, b_t = pts(arrival_profile, traffic_signal_state)
    est_queue_data = [x_t[t] for t in range(1, len(arrival_profile) + 1)]
    
    queue_length_distribution = calculate_queue_length_distribution(data, cycle_length, delta_u)
    sim_queue_data = [probs for time_point, probs in sorted(queue_length_distribution.items())]
    
    percentage_error = calculate_error(est_queue_data, sim_queue_data)
    
    phis = np.arange(0.01, 0.301, 0.001)
    mpe_values = []
    
    for phi in phis:
        mpe_list = []
        for _ in range(30):
            arrival_profile = calculate_arrival_profile(data, cycle_length, delta_u, phi)
            traffic_signal_state = calculate_traffic_signal_state(cycle_length, green_start, green_end)
            x_t, b_t = pts(arrival_profile, traffic_signal_state)
            est_queue_data = [x_t[t] for t in range(1, len(arrival_profile) + 1)]
            queue_length_distribution = calculate_queue_length_distribution(data, cycle_length, delta_u)
            sim_queue_data = [probs for time_point, probs in sorted(queue_length_distribution.items())]
            mpe = calculate_error(est_queue_data, sim_queue_data)
            mpe_list.append(mpe)
        mpe_values.append(np.mean(mpe_list))
    
    min_error = np.min(mpe_values)
    min_phi = phis[np.argmin(mpe_values)]
    
    return phis, mpe_values, min_phi, min_error

file_path1 = 'C:/Users/john/Documents/over.csv'
file_path2 = 'C:/Users/john/Documents/0.95.csv'
file_path3 = 'C:/Users/john/Documents/0.7.csv'

# First set of parameters for file 7600.csv
cycle_length1 = 60
green_start1 = 33
green_end1 = 53
delta_u1 = 4
phi1 = 0.05

# Second set of parameters for file 4500.csv
cycle_length2 = 60
green_start2 = 33
green_end2 = 53
delta_u2 = 4
phi2 = 0.05

# Third set of parameters for file 1600.csv
cycle_length3 = 60
green_start3 = 33
green_end3 = 53
delta_u3 = 4
phi3 = 0.05

# Process data for the first file with first set of parameters
phis1, mpe_values1, min_phi1, min_error1 = process_data(file_path1, cycle_length1, green_start1, green_end1, delta_u1, phi1)

# Process data for the second file with second set of parameters
phis2, mpe_values2, min_phi2, min_error2 = process_data(file_path2, cycle_length2, green_start2, green_end2, delta_u2, phi2)

# Process data for the second file with second set of parameters
phis3, mpe_values3, min_phi3, min_error3 = process_data(file_path3, cycle_length3, green_start3, green_end3, delta_u3, phi3)

plt.figure(figsize=(10, 6))

# Plot for the first file and parameters
plt.plot(phis1, mpe_values1, marker='o', label='7600 vph Percentage Error')
plt.scatter([min_phi1], [min_error1], color='red', zorder=5)
plt.text(min_phi1, min_error1, f'Error\nφ={min_phi1:.3f}\nError={min_error1:.4f}', fontsize=9, ha='left')

# Plot for the second file and parameters
plt.plot(phis2, mpe_values2, marker='s', label='4500 vph Percentage Error')
plt.scatter([min_phi2], [min_error2], color='blue', zorder=5)
plt.text(min_phi2, min_error2, f'Error\nφ={min_phi2:.3f}\nError={min_error2:.4f}', fontsize=9, ha='left')

# Plot for the third file and parameters
plt.plot(phis3, mpe_values3, marker='s', label='1600 vph Percentage Error')
plt.scatter([min_phi3], [min_error3], color='green', zorder=5)
plt.text(min_phi3, min_error3, f'Error\nφ={min_phi2:.3f}\nError={min_error3:.4f}', fontsize=9, ha='left')

plt.xlabel('Penetration Rate % $\phi$')
plt.ylabel('Percentage Error %')
plt.title('Percentage Error % vs Penetration Rate $\phi$')
plt.legend()
plt.show()
