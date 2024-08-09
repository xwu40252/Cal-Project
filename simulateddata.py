# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 08:58:11 2024

@author: Nierrr
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def calculate_traffic_signal_state(cycle_length, green_start, green_end):
    traffic_signal_state = [0] * cycle_length
    for i in range(green_start - 1, green_end):
        if i < cycle_length:
            traffic_signal_state[i] = 1
    return traffic_signal_state

def calculate_arrival_profile(data, cycle_length, delta_u, phi):
    data = data[['Simulation Time', 'Vehicle Number', 'Position', 'Speed']].dropna()
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
            queue_length = np.sum(time_data['Speed'] <= 5) * delta_u
            queue_lengths[time_in_cycle + cycle * cycle_length].append(queue_length)
    
    queue_length_distribution = {}
    max_length = max(max(lengths) for lengths in queue_lengths.values())
    for time_point, lengths in queue_lengths.items():
        length_counts = pd.Series(lengths).value_counts(normalize=True).reindex(np.arange(max_length+1), fill_value=0)
        queue_length_distribution[time_point] = length_counts.tolist()
    
    return queue_lengths, queue_length_distribution

files_and_params = [
    {
        'path': 'C:/Users/john/Documents/over.xlsx',
        'cycle_length': 60,
        'green_start': 33,
        'green_end': 53,
        'delta_u': 0.6,
        'phi': 0.6
    },
    {
        'path': 'C:/Users/john/Documents/0.95.xlsx',
        'cycle_length': 60,
        'green_start': 33,
        'green_end': 53,
        'delta_u': 0.583,
        'phi': 0.6
    },
    {
        'path': 'C:/Users/john/Documents/0.7.xlsx',
        'cycle_length': 60,
        'green_start': 33,
        'green_end': 53,
        'delta_u': 0.43,
        'phi': 0.6
    },
    {
        'path': 'C:/Users/john/Documents/0.4.xlsx',
        'cycle_length': 60,
        'green_start': 33,
        'green_end': 53,
        'delta_u': 0.2389,
        'phi': 0.6
    }
]

all_max_lengths = []

for file_info in files_and_params:
    data = pd.read_excel(file_info['path'])
    
    filtered_data = data[data['Lane'] == '1-3'].dropna(subset=['Lane', 'Simulation Time', 'Speed'])
    
    
    repetitions = 10
    max_lengths_list = []

    for _ in range(repetitions):
        max_lengths = []
        queue_lengths, _ = calculate_queue_length_distribution(filtered_data, file_info['cycle_length'], file_info['delta_u'])
        
        for i in range(0, 600, 60):
            interval_max_length = 0
            for j in range(60):
                time_point = i + j
                if time_point in queue_lengths:
                    interval_max_length = max(interval_max_length, max(queue_lengths[time_point]))
            max_lengths.append(interval_max_length)
        
        max_lengths_list.append(max_lengths)
    
    avg_max_lengths = np.mean(max_lengths_list, axis=0)
    all_max_lengths.append({
        'file_name': file_info['path'].split('/')[-1],
        'max_lengths': avg_max_lengths
    })

plt.figure(figsize=(12, 8))

labels = ['over saturated', 'CS = 0.95', 'CS = 0.7', 'CS = 0.4']

for idx, max_lengths in enumerate(all_max_lengths):
    time_points = np.arange(0, 600, 60)
    plt.plot(time_points, max_lengths['max_lengths'], marker='o', label=labels[idx])

plt.xlabel('Time (s)')
plt.ylabel('Maximum Queue Length (Numbers of vehicles)')
plt.title('Simulated Maximum Queue Length vs Time')
plt.legend()
plt.grid(True)
plt.show()
