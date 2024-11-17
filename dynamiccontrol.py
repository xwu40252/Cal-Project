# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:19:34 2024

@author: Nierrr
"""

import win32com.client as com
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Vissim = com.Dispatch("Vissim.Vissim.2400")

Path_of_COM_Basic_Commands_network = r"C:\Users\zhexi\OneDrive\Desktop\LB"
network_filename = 'video.inpx'
layout_filename = 'video.layx'

network_filepath = os.path.join(Path_of_COM_Basic_Commands_network, network_filename)
layout_filepath = os.path.join(Path_of_COM_Basic_Commands_network, layout_filename)

Vissim.LoadNet(network_filepath, False)
Vissim.LoadLayout(layout_filepath)

Random_Seed = 42
Vissim.Simulation.SetAttValue('RandSeed', Random_Seed)

End_of_simulation = 600
Vissim.Simulation.SetAttValue('SimPeriod', End_of_simulation)

signal_controllers = {
    'A': {'SC': 1, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 1, 'WE_QC': 2, 'NS_Link': 90, 'WE_Link': 87},
    'B': {'SC': 2, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 4, 'WE_QC': 5, 'NS_Link': 2, 'WE_Link': 86},
    'C': {'SC': 3, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 8, 'WE_QC': 10, 'NS_Link': 6, 'WE_Link': 78},
    'D': {'SC': 4, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 13, 'WE_QC': 12, 'NS_Link': 72, 'WE_Link': 76},
    'E': {'SC': 5, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 15, 'WE_QC': 17, 'NS_Link': 18, 'WE_Link': 70},
    'F': {'SC': 6, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 18, 'WE_QC': 19, 'NS_Link': 16, 'WE_Link': 69},
    'G': {'SC': 7, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 22, 'WE_QC': 25, 'NS_Link': 19, 'WE_Link': 62},
    'H': {'SC': 8, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 25, 'WE_QC': 29, 'NS_Link': 24, 'WE_Link': 58},
    'I': {'SC': 9, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 32, 'WE_QC': 33, 'NS_Link': 34, 'WE_Link': 54},
    'J': {'SC': 10, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 36, 'WE_QC': 35, 'NS_Link': 33, 'WE_Link': 49},
    'K': {'SC': 11, 'NS_SG': 1, 'WE_SG': 2, 'NS_QC': 40, 'WE_QC': 41, 'NS_Link': 41, 'WE_Link': 42}
}

SignalControllerCycleTime = 60
for intersection, info in signal_controllers.items():
    SC = Vissim.Net.SignalControllers.ItemByKey(info['SC'])
    SC.SetAttValue('CycTm', SignalControllerCycleTime)


phi_list = [0.3]

def pts(arrival_profile, traffic_signal_state, delta_u):
    a = [0] + arrival_profile
    s = [0] + traffic_signal_state
    T = len(a)
    max_queue = 35

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
    
    q_t = q_t * delta_u

    return x_t.tolist(), b_t.tolist(), q_t.tolist()

def calculate_traffic_signal_state(cycle_length, green_start, green_end):
    traffic_signal_state = [0] * cycle_length
    for i in range(green_start - 1, green_end):
        if i < cycle_length:
            traffic_signal_state[i] = 1
    return traffic_signal_state

def calculate_arrival_profile(data, cycle_length, delta_u, phi, direction_link):
    data = data[data['Lane'].str.contains(f'^{direction_link}-')]

    if data.empty:
        return [0] * cycle_length

    data = data[['Simulation Time', 'Vehicle Number', 'Position', 'Speed']]
    N_c = int(data['Simulation Time'].max() // cycle_length) + 1
    unique_vehicle_numbers = data['Vehicle Number'].unique()
    selected_vehicles = np.random.choice(unique_vehicle_numbers, size=int(phi * len(unique_vehicle_numbers)), replace=False)
    sampled_data = data[data['Vehicle Number'].isin(selected_vehicles)]
    sampled_data = sampled_data.sort_values(by=['Vehicle Number', 'Simulation Time']).drop_duplicates(subset=['Vehicle Number'], keep='first')
    sampled_data['TimeInCycle'] = sampled_data['Simulation Time'] % cycle_length
    arrivals_per_time_point = sampled_data.groupby('TimeInCycle')['Vehicle Number'].nunique().reindex(np.arange(cycle_length), fill_value=0)
    arrival_profile = (arrivals_per_time_point / (N_c * delta_u * phi)).tolist()
    
    arrival_profile = [0 if np.isnan(x) else x for x in arrival_profile]
    return arrival_profile

delta_u = 1

queue_counter_results = {'Time': []}
for i in range(1, 21):
    queue_counter_results[f'QC{i}'] = []

for phi in phi_list:
    print(f"Testing with penetration rate: {phi}")
    
    estimated_queue_length_data = {f"{intersection}_{direction}": [] for intersection in signal_controllers.keys() for direction in ['NS', 'WE']}
    offsets = {}
    green_bandwidths = []
    total_time = 0

    for intersection, info in signal_controllers.items():
        for direction in ['NS', 'WE']:
            SG = Vissim.Net.SignalControllers.ItemByKey(info['SC']).SGs.ItemByKey(info[f'{direction}_SG'])
            distance = Vissim.Net.Links.ItemByKey(info[f'{direction}_Link']).AttValue('Length2D')
            arrival_time = total_time + (distance / 36.45)
            time_at_intersection = arrival_time % SignalControllerCycleTime
            green_start = SG.AttValue('EndRed') % SignalControllerCycleTime
            green_interval = SG.AttValue('EndGreen') - SG.AttValue('EndRed')
            green_end = (green_start + green_interval) % SignalControllerCycleTime

            green_start = 0 if pd.isna(green_start) else int(green_start)
            green_end = 0 if pd.isna(green_end) else int(green_end)

            if time_at_intersection < green_start:
                offset = green_start - time_at_intersection
            else:
                offset = (SignalControllerCycleTime - time_at_intersection + green_start) % SignalControllerCycleTime

            offsets[f'{intersection}_{direction}'] = offset
            total_time = arrival_time + offset
            green_bandwidths.append((green_start, green_end))

            print(f"Offset at {intersection}_{direction}: {offset:.2f} seconds")

    max_green_bandwidth = SignalControllerCycleTime
    for i in range(len(green_bandwidths) - 1):
        overlap_start = max(green_bandwidths[i][0], green_bandwidths[i + 1][0])
        overlap_end = min(green_bandwidths[i][1], green_bandwidths[i + 1][1])
        green_bandwidth = max(0, overlap_end - overlap_start)
        max_green_bandwidth = min(max_green_bandwidth, green_bandwidth)

    print(f"Maximum green bandwidth: {max_green_bandwidth:.2f} seconds")
    print(f"Offsets at each intersection: {offsets}")
    print("------------------------------------------------------")

    try:
        for t in range(0, End_of_simulation, 180):
            print(f"Simulation time: {t} to {t + 180} seconds with phi={phi}")
            
            vehicle_data = []
            for _ in range(SignalControllerCycleTime):
                Vissim.Simulation.RunSingleStep()
                All_Vehicles = Vissim.Net.Vehicles.GetAll()
                for vehicle in All_Vehicles:
                    simulation_time = Vissim.Simulation.SimulationSecond
                    veh_number = vehicle.AttValue('No')
                    veh_type = vehicle.AttValue('VehType')
                    veh_speed = vehicle.AttValue('Speed')
                    veh_position = vehicle.AttValue('Pos')
                    veh_linklane = vehicle.AttValue('Lane')
                    vehicle_data.append([simulation_time, veh_number, veh_type, veh_speed, veh_position, veh_linklane])

            vehicle_df = pd.DataFrame(vehicle_data, columns=['Simulation Time', 'Vehicle Number', 'Vehicle Type', 'Speed', 'Position', 'Lane'])

            queue_lengths = {}
            total_queue_lengths = {intersection: 0 for intersection in signal_controllers.keys()}
            
            for intersection, info in signal_controllers.items():
                for direction in ['NS', 'WE']:
                    SG = Vissim.Net.SignalControllers.ItemByKey(info['SC']).SGs.ItemByKey(info[f'{direction}_SG'])
                    green_start = int(SG.AttValue('EndRed') if not pd.isna(SG.AttValue('EndRed')) else 0)
                    green_end = int(SG.AttValue('EndGreen') if not pd.isna(SG.AttValue('EndGreen')) else 0)
                    cycle_length = SignalControllerCycleTime
                    traffic_signal_state = calculate_traffic_signal_state(cycle_length, green_start, green_end)
                    
                    direction_link = info[f'{direction}_Link']
                    arrival_profile = calculate_arrival_profile(vehicle_df, cycle_length, delta_u, phi, direction_link)
                    
                    _, _, estimated_queue_lengths = pts(arrival_profile, traffic_signal_state, delta_u)
                    max_queue_length = max(estimated_queue_lengths)
                    estimated_queue_length_data[f"{intersection}_{direction}"].append(max_queue_length)
                    queue_lengths[f"{intersection}_{direction}"] = max_queue_length
                    total_queue_lengths[intersection] += max_queue_length

            green_splits = {}
            for intersection in signal_controllers.keys():
                if total_queue_lengths[intersection] > 0:
                    green_splits[f'{intersection}_NS'] = queue_lengths[f'{intersection}_NS'] / total_queue_lengths[intersection]
                    green_splits[f'{intersection}_WE'] = queue_lengths[f'{intersection}_WE'] / total_queue_lengths[intersection]
                else:
                    green_splits[f'{intersection}_NS'] = 1 / 2
                    green_splits[f'{intersection}_WE'] = 1 / 2

            green_intervals = {}
            for direction, split in green_splits.items():
                green_intervals[direction] = SignalControllerCycleTime * split
                
                if green_intervals[direction] > 55:
                    green_intervals[direction] = 55
                    
                print(f"Greeninterval in {direction}: {green_intervals[direction]:.2f} seconds")

            for intersection, info in signal_controllers.items():
                for direction in ['NS', 'WE']:
                    SG = Vissim.Net.SignalControllers.ItemByKey(info['SC']).SGs.ItemByKey(info[f'{direction}_SG'])
                    if queue_lengths[f"{intersection}_{direction}"] == 0:
                        print(f"Queue length in {intersection}_{direction} is zero, keeping the previous cycle's signal state")
                        continue
                    else:
                        current_endgreen = SG.AttValue('EndGreen')
                        current_endred = SG.AttValue('EndRed')
                        new_greeninterval = green_intervals[f'{intersection}_{direction}']
            
                        new_endgreen = current_endgreen
                        new_endred = current_endgreen - new_greeninterval

                        min_green = SG.AttValue('MinGreen')
                        max_green = SG.AttValue('MaxGreen')
                        min_red = SignalControllerCycleTime - max_green
                        max_red = SignalControllerCycleTime - min_green

                        if new_endgreen > 55:
                            difference = new_endgreen - 55
                            new_endgreen = 55
                            new_endred = new_endred - difference
                            
                        if new_endgreen > max_green:
                            new_endgreen = max_green
                        elif new_endgreen < min_green:
                            new_endgreen = min_green

                        if new_endred > max_red:
                            new_endred = max_red
                        elif new_endred < min_red:
                            new_endred = min_red

                        SG.SetAttValue('EndGreen', new_endgreen)
                        SG.SetAttValue('EndRed', new_endred)

                        print(f"Updated endgreen for {intersection}_{direction}: {new_endgreen:.2f} seconds, endred: {new_endred:.2f} seconds")

            for i, qc_key in enumerate(queue_counter_results.keys()):
                if i == 0:
                    queue_counter_results['Time'].append(t)
                else:
                    QC = Vissim.Net.QueueCounters.ItemByKey(i)
                    queue_counter_results[qc_key].append(QC.AttValue("QLenMax(Current, Last)"))

        print(f"Estimated queue lengths for phi={phi}: {estimated_queue_length_data}")
        print("------------------------------------------------------")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        Vissim.Simulation.Stop()
