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



def main():
    # sampling time / step size
    delta_T = 0.1
    
    #Initial states (x_position,y_position,x_velocity,y_velocity)
    initial_state = np.array([8.7,53.3,5,np.pi/2])
    
    # Initialize a kalman filter instance
    KF = Kalman_Filter(initial_state,delta_T)
    
    # Steering angle of front wheel
    steering = np.pi/100
    
    # 'Throttle' acceleration units in m/s^2
    longitudinal_acceleration = 0

    KF.current_inputs = [longitudinal_acceleration,steering]

    prediction_time = 6
    future_states = KF.predict_plot_future(prediction_time, plot = False)
    safe_states,collision_time = KF.find_collision_state(future_states)
    print('Collision Time : ' + str(collision_time))

    KF.plot_track(safe_states)


    # update in real time
    # for i in range(0,10):
    #     KF.update(None,[lateral_acceleration,longitudinal_acceleration])
    
    # states = KF.plot_predicted_path_at_current_state(prediction_time, True)

        

if __name__ == "__main__":
    main()

    
