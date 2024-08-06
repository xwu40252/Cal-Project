# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:01:55 2024

@author: Nierrr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Vehicle length in feet
VEHICLE_LENGTH = 15

def pts(arrival_profile, traffic_signal_state, delta_u, initial_queue_length):
    a = [0] + arrival_profile
    s = [0] + traffic_signal_state
    T = len(a)
    max_queue = 11  # Change the maximum queue length to 5

    x_t = np.zeros((T, max_queue))
    b_t = np.zeros(T)
    q_t = np.zeros(T)  # Initialize queue length array

    # Set the initial queue length
    if initial_queue_length < max_queue:
        x_t[0][initial_queue_length] = 1.0
        q_t[0] = initial_queue_length  # Initialize q_t with initial queue length
    else:
        x_t[0][max_queue - 1] = 1.0
        q_t[0] = max_queue - 1

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
        
        if s[t] == 0:
            q_t[t] = q_t[t-1] + a[t] * (1 - np.sum(x_t[t-1][1:]))
        else:
            q_t[t] = max(q_t[t-1] - b_t[t], 0)
    
    # Multiply q_t by delta_u
    q_t = q_t * delta_u * VEHICLE_LENGTH

    return x_t.tolist(), b_t.tolist(), q_t.tolist()

def calculate_traffic_signal_state(cycle_length, green_start, green_end):
    traffic_signal_state = [0] * cycle_length
    for i in range(green_start - 1, green_end):
        if i < cycle_length:
            traffic_signal_state[i] = 1
    return traffic_signal_state

def calculate_arrival_profile(data, cycle_length, delta_u, phi):
    # Filter data for vehicles in lanes labeled as "1-3"
    data = data[data['Lane'] == '1-3']
    
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
    data = data.dropna(subset=['Simulation Time', 'Speed', 'Lane'])
    data['Simulation Time'] = data['Simulation Time'] - 2
    data['TimeInCycle'] = data['Simulation Time'] % cycle_length
    data['Cycle'] = data['Simulation Time'] // cycle_length

    if data.empty:
        print("No data available after dropping NaN values.")
        return {}, {}

    max_cycle = int(data['Cycle'].max())
    print(f"Max cycle calculated: {max_cycle}")
    
    queue_lengths = defaultdict(list)
    
    for cycle in range(max_cycle + 1):
        cycle_data = data[data['Cycle'] == cycle]
        for time_in_cycle in range(cycle_length):
            time_data = cycle_data[cycle_data['TimeInCycle'] == time_in_cycle]
            queue_length = np.sum(time_data['Speed'] <= 5) * delta_u * VEHICLE_LENGTH
            queue_lengths[time_in_cycle + cycle * cycle_length].append(queue_length)
    
    queue_length_distribution = {}
    max_length = max(max(lengths) for lengths in queue_lengths.values())
    for time_point, lengths in queue_lengths.items():
        length_counts = pd.Series(lengths).value_counts(normalize=True).reindex(np.arange(max_length+1), fill_value=0)
        queue_length_distribution[time_point] = length_counts.tolist()
    
    return queue_lengths, queue_length_distribution

# Fixed phi value
phi = 0.6

# Parameters for the '0.95.xlsx' file
file_info = {
    'path': 'C:/Users/john/Documents/0.95.xlsx',
    'cycle_length': 60,
    'green_start': 33,
    'green_end': 53,
    'delta_u': 1
}

# Define labels for the plot
label = 'over'

# Load the data
data = pd.read_excel(file_info['path'])

# Plot the maximum queue length for the '0.95.xlsx' file
plt.figure(figsize=(12, 8))

# Simulated data
q_t_max_values_initial_0 = []
q_t_max_values_initial_max = []

for _ in range(40):
    arrival_profile = calculate_arrival_profile(data, file_info['cycle_length'], file_info['delta_u'], phi)
    traffic_signal_state = calculate_traffic_signal_state(file_info['cycle_length'], file_info['green_start'], file_info['green_end'])
    
    # Run simulation with initial queue length 0
    _, _, q_t_initial_0 = pts(arrival_profile, traffic_signal_state, file_info['delta_u'], 0)
    q_t_max_values_initial_0.append(q_t_initial_0[:10])  # 前600s的每60s的队列长度
    
    # Run simulation with initial queue length at maximum capacity
    _, _, q_t_initial_max = pts(arrival_profile, traffic_signal_state, file_info['delta_u'], 10)
    q_t_max_values_initial_max.append(q_t_initial_max[:10])  # 前600s的每60s的队列长度

q_t_max_values_initial_0 = np.max(q_t_max_values_initial_0, axis=0)
q_t_max_values_initial_max = np.max(q_t_max_values_initial_max, axis=0)

cycle_time_points = np.arange(len(q_t_max_values_initial_0))
plt.plot(cycle_time_points * 60, q_t_max_values_initial_0, marker='o', label="Queue estimate with zero initial condition", linewidth=3)
plt.plot(cycle_time_points * 60, q_t_max_values_initial_max, marker='x', label="Queue estimate with saturated initial condition", linewidth=3)

# Add numerical values on the plots
for i, (x, y) in enumerate(zip(cycle_time_points * 60, q_t_max_values_initial_0)):
    plt.text(x, y, f'({x}, {y:.1f})', fontsize=8, ha='center', va='bottom')

for i, (x, y) in enumerate(zip(cycle_time_points * 60, q_t_max_values_initial_max)):
    plt.text(x, y, f'({x}, {y:.1f})', fontsize=8, ha='center', va='bottom')

# Measured data
# Update the file info for measured data
file_info['delta_u'] = 0.583

# Filter for Link/Lane '1-3'
filtered_data = data[data['Lane'] == '1-3'].dropna(subset=['Lane', 'Simulation Time', 'Speed'])

repetitions = 20  # Repeat 20 times to reduce error
max_lengths_list = []

for _ in range(repetitions):
    max_lengths = []
    queue_lengths, _ = calculate_queue_length_distribution(filtered_data, file_info['cycle_length'], file_info['delta_u'])
    
    # Calculate max lengths for each 60-second interval up to 600 seconds
    for i in range(0, 600, 60):  # 600 seconds with 60-second intervals
        interval_max_length = 0
        for j in range(60):  # Calculate within the 60-second interval
            time_point = i + j
            if time_point in queue_lengths:
                interval_max_length = max(interval_max_length, max(queue_lengths[time_point]))
        max_lengths.append(interval_max_length)
    
    max_lengths_list.append(max_lengths)

# Average max lengths over all repetitions
avg_max_lengths = np.mean(max_lengths_list, axis=0)

# Plotting measured data
time_points = np.arange(0, 600, 60)
plt.plot(time_points, avg_max_lengths, marker='x', label='Actual queue length', linewidth=3)

# Add numerical values on the plots for measured data
for i, (x, y) in enumerate(zip(time_points, avg_max_lengths)):
    plt.text(x, y, f'({x}, {y:.1f})', fontsize=8, ha='center', va='bottom')

# Final plot adjustments
plt.xlabel('Time (s)', fontsize=30)  
plt.ylabel('Queue length (ft)', fontsize=30)  
plt.xticks(fontsize=24) 
plt.yticks(fontsize=24)    
plt.grid(True)
plt.legend(fontsize=20)
plt.show()
