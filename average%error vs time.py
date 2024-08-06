# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:14:33 2024

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
    data['Simulation Time'] = data['Simulation Time'] - 2
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
    data['Simulation Time'] = data['Simulation Time'] - 2
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
        percentage_error = np.abs(((est_queue_data_np - sim_queue_data_np[:, :est_queue_data_np.shape[1]]) / sim_queue_data_np[:, :est_queue_data_np.shape[1]])) * 100
        percentage_error = np.nan_to_num(percentage_error, nan=0.0, posinf=0.0, neginf=0.0)
    return percentage_error

def process_and_plot(file_path, label, delta_u, phi, repetitions=10):
    data = pd.read_excel(file_path)
    
    average_errors = np.zeros(10)
    
    for _ in range(repetitions):
        arrival_profile = calculate_arrival_profile(data, cycle_length, delta_u, phi)
        traffic_signal_state = calculate_traffic_signal_state(cycle_length, green_start, green_end)

        x_t, b_t = pts(arrival_profile, traffic_signal_state)
        est_queue_data = [x_t[t] for t in range(1, len(arrival_profile) + 1)]

        queue_length_distribution = calculate_queue_length_distribution(data, cycle_length, delta_u)
        sim_queue_data = [probs for time_point, probs in sorted(queue_length_distribution.items())]

        percentage_errors = []
        for t in range(1, 601, 60):
            if t // 60 < len(sim_queue_data) and len(est_queue_data) > 0:
                est_data = [est_queue_data[t // 60]]
                sim_data = [sim_queue_data[t // 60]]
                error = calculate_error(est_data, sim_data)
                percentage_errors.append(np.mean(error))
        
        average_errors += np.array(percentage_errors)
    
    average_errors /= repetitions
    
    plt.plot(range(60, 601, 60), average_errors, marker='o', label=f'{label} - φ={phi}')

cycle_length = 60
green_start = 33
green_end = 53

datasets = [
    ('C:/Users/john/Documents/0.95.xlsx', '0.95', 4),
    ('C:/Users/john/Documents/over.xlsx', 'Over', 4),
    ('C:/Users/john/Documents/0.7.xlsx', '0.7', 4),
    ('C:/Users/john/Documents/0.4.xlsx', '0.4', 4)
]

phi_values = [0.1, 0.2, 0.3]

plt.figure(figsize=(10, 6))

for file_path, label, delta_u in datasets:
    for phi in phi_values:
        process_and_plot(file_path, label, delta_u, phi)

plt.xlabel('Time (s)')
plt.ylabel('Percentage Error %')
plt.title('Average Percentage Error % vs Time (s) for Different Datasets and φ Values')
plt.legend()
plt.grid(True)
plt.show()
