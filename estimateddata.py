import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pts(arrival_profile, traffic_signal_state, delta_u):
    a = [0] + arrival_profile
    s = [0] + traffic_signal_state
    T = len(a)
    max_queue = 11

    x_t = np.zeros((T, max_queue))
    b_t = np.zeros(T)
    q_t = np.zeros(T)
    
    initial_queue_length = np.random.randint(0, max_queue)
    x_t[0][initial_queue_length] = 1.0

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
    
    q_t = q_t * delta_u * 15

    return x_t.tolist(), b_t.tolist(), q_t.tolist()

def calculate_traffic_signal_state(cycle_length, green_start, green_end):
    traffic_signal_state = [0] * cycle_length
    for i in range(green_start - 1, green_end):
        if i < cycle_length:
            traffic_signal_state[i] = 1
    return traffic_signal_state

def calculate_arrival_profile(data, cycle_length, delta_u, phi):
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

files_and_params = [
    {
        'path': 'C:/Users/john/Documents/over.xlsx',
        'cycle_length': 60,
        'green_start': 33,
        'green_end': 53,
        'delta_u': 1
    },
    {
        'path': 'C:/Users/john/Documents/0.95.xlsx',
        'cycle_length': 60,
        'green_start': 33,
        'green_end': 53,
        'delta_u': 1
    },
    {
        'path': 'C:/Users/john/Documents/0.7.xlsx',
        'cycle_length': 60,
        'green_start': 33,
        'green_end': 53,
        'delta_u': 1
    },
    {
        'path': 'C:/Users/john/Documents/0.4.xlsx',
        'cycle_length': 60,
        'green_start': 33,
        'green_end': 53,
        'delta_u': 1
    },
]

# Fixed phi value
phi = 0.6

# Define labels for each file
labels = {
    'over.xlsx': 'over saturated',
    '0.95.xlsx': 'CS = 0.95',
    '0.7.xlsx': 'CS = 0.7',
    '0.4.xlsx': 'CS = 0.4'
}

# 绘制每个文件的最大队列长度
plt.figure(figsize=(12, 8))

for file_info in files_and_params:
    data = pd.read_excel(file_info['path'])
    
    q_t_max_values = []

    for _ in range(40):
        arrival_profile = calculate_arrival_profile(data, file_info['cycle_length'], file_info['delta_u'], phi)
        traffic_signal_state = calculate_traffic_signal_state(file_info['cycle_length'], file_info['green_start'], file_info['green_end'])
        x_t, b_t, q_t = pts(arrival_profile, traffic_signal_state, file_info['delta_u'])
        q_t_max_values.append(q_t[:10]) 

    q_t_max_values = np.max(q_t_max_values, axis=0)
    
    label = f"{labels[file_info['path'].split('/')[-1]]})"
    cycle_time_points = np.arange(len(q_t_max_values))
    plt.plot(cycle_time_points * 60, q_t_max_values, marker='o', label=label) 

plt.xlabel('Time (s)')
plt.ylabel('Queue Length (Number of vehicles)')
plt.title('Estimated Maximum Queue Length vs Time')
plt.legend()
plt.grid(True)
plt.show()
