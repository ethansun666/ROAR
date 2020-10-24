# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:02:19 2020

@author: Ethan
"""

from ROAR.kalman_filter.KalmanFilter import Kalman_Filter
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import json

def read_map():
    f = open('ROAR/kalman_filter/easy_map_waypoints_pointcloud_v3.json')
    data = json.load(f)
    a = []
    b = []
    for point in data :
        a.append(point['point_a'])
        b.append(point['point_b'])
    a = np.asarray(a)
    b = np.asarray(b)
    # only use x and y here
    a = a[:,0:2]
    b = b[:,0:2]
    b_filtered = []
    for point in b:
        if point[0] < 600 and point[0] > -600 and point[1] < 600 and point[1] > -600:
            b_filtered.append(point)
    b_filtered = np.asarray(b_filtered)
    return a,b_filtered

def plot_track(KF):
    a,b = read_map()
    
    prediction_time = 3
    states = KF.predict_plot_future(prediction_time, plot = False)

    fig, ax = KF.plot_path(states)
    
    ax.plot(a[:,0],a[:,1])
    ax.plot(b[:,0],b[:,1])
    
    ax.set_xlim(-30,30)
    ax.set_ylim(40,90)


def main():
    # sampling time / step size
    delta_T = 0.1
    
    #Initial states (x_position,y_position,x_velocity,y_velocity)
    initial_state = np.array([8.68,53.25,0,10])
    
    # Initialize a kalman filter instance
    KF = Kalman_Filter(initial_state,delta_T)
    
    # 'Steering' acceleration units in m/s^2
    lateral_acceleration = -2
    
    # 'Throttle' acceleration units in m/s^2
    longitudinal_acceleration = 0
    
    KF.system_inputs = [lateral_acceleration,longitudinal_acceleration]
    
    plot_track(KF)
    
    # prediction_time = 5
    
    # states = KF.predict_plot_future(prediction_time, plot = True)
    
    # for i in range(0,10):
    #     KF.update(None,[lateral_acceleration,longitudinal_acceleration])
    
    # states = KF.plot_predicted_path_at_current_state(prediction_time, True)

        

if __name__ == "__main__":
    main()

    
