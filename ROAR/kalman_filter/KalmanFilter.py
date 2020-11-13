# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:12:46 2020

@author: Ethan
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from filterpy.kalman import ExtendedKalmanFilter
import json
import time

class Kalman_Filter:
    # state variables (x_position,y_position,x_velocity,y_velocity)
    # state variables (x_position,y_position,scalar_speed,global_heading_angle)
    current_state : np.array((4,)) 
    # state inputs (x_acceleration,y_acceleration (refers to steering & throttle))
    # state inputs (longitudinal_acceleration, steering_angle (refers to steering & throttle))
    current_inputs : np.array((2,))
    predicted_state_current : np.array((4,))
    delta_T : float
    # System matrix is not applicable in nonlinear case
    # SYSTEM_MATRIX : np.array((4,4))     #State Transition Matrix
    # INPUT_MATRIX : np.array((4,2))
    COLLISION_THRESHOLD_DIS : float
    PREDICTION_TIME : float
    Lf : float
    Lr : float

    
    def __init__(self,initial_state,delta_T):
        self.current_state = initial_state
        self.predicted_state_current = self.current_state
        self.delta_T = delta_T
        # self.SYSTEM_MATRIX = np.array([[1,0,delta_T,0],[0,1,0,delta_T],[0,0,1,0],[0,0,0,1]])
        # self.INPUT_MATRIX = np.array([[0.5*delta_T**2,0],[0,0.5*delta_T**2],[delta_T,0],[0,delta_T]])
        self.COLLISION_THRESHOLD_DIS = 5
        self.Lf = 1.7
        self.Lr = 1.7
        self.map = self.__read_map()

        
    def predict_one_step(self,state,inputs):
        theta = state[3]
        v_x = np.cos(theta)*state[2]
        v_y = np.sin(theta)*state[2]

        # print('theta : '+str(theta*360/(2*np.pi)))
        # inputs = self.__carfixed_to_xy(inputs_carfixed,theta)
        # predicted_next_state = self.SYSTEM_MATRIX@state + self.INPUT_MATRIX@inputs
        # self.__print_state(predicted_next_state)
        predicted_next_state = np.zeros(4,)
        beta = np.arctan(np.tan(inputs[1])*self.Lr/(self.Lf+self.Lr))
        predicted_next_state[0] = state[0]+self.delta_T*(state[2]*np.cos(state[3]+beta))
        predicted_next_state[1] = state[1]+self.delta_T*(state[2]*np.sin(state[3]+beta))
        predicted_next_state[2] = state[2]+self.delta_T*inputs[0]
        predicted_next_state[3] = state[3]+self.delta_T*state[2]*np.sin(beta)/self.Lr
        self.predicted_state_current = predicted_next_state
        return self.predicted_state_current
    
    def update(self,measurements,inputs):
        self.current_inputs = inputs
        predicted_state = self.predict_one_step(self.current_state, inputs)
        self.current_state = predicted_state
        if measurements != None:
            self.current_state[0] = measurements[0]
            self.current_state[1] = measurements[1]
            self.current_state[2] = measurements[2]
            self.current_state[3] = measurements[3]

        
    def predict_plot_future(self,prediction_time, plot=False):
        start = time.time()
        self.predicted_state_current = self.current_state
        states=[]
        states.append(self.current_state)
        for i in range(1,int(prediction_time/self.delta_T)):
            next_state = self.predict_one_step(self.predicted_state_current,self.current_inputs)
            states.append(next_state)
        states = np.asarray(states)
        if plot:    
            fig,ax= self.__plot_path(states)
        end = time.time()
        # print('predict_plot_future : ' + str(end-start) + ' seconds')
        return states

    def find_collision_state(self,states):
        start = time.time()
        left, right = self.map[0],self.map[1]
        collision = False
        safe_states = []
        for state in states:
            pos = [state[0], state[1]]
            for a in left:
                if self.__euclidean_dis(a, pos) <= self.COLLISION_THRESHOLD_DIS:
                    # print('Collision Detected! Distance : ' + str(self.__euclidean_dis(a, pos)))
                    collision = True
                    break
            for b in right:
                if self.__euclidean_dis(b, pos) <= self.COLLISION_THRESHOLD_DIS:
                    # print('Collision Detected! Distance : ' + str(self.__euclidean_dis(b, pos)))
                    collision = True
                    break
            if collision:
                break
            safe_states.append(state)
        safe_states = np.asarray(safe_states)
        if collision:
            collision_time = len(safe_states) * 0.1
        else:
            collision_time = None
        end = time.time()
        # print('find_collision_state : ' + str(end-start) + ' seconds')
        return safe_states, collision_time

    def __carfixed_to_xy(self,vector_carfixed,theta):
        vector_xy = np.zeros(2,)
        vector_xy[0] = vector_carfixed[0]*np.cos(theta)+vector_carfixed[1]*np.sin(theta)
        vector_xy[1] = -vector_carfixed[0]*np.sin(theta)+vector_carfixed[1]*np.cos(theta)
        # print(vector_xy)
        return vector_xy

    def __euclidean_dis(self,a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def __read_map(self):
        f = open('ROAR/kalman_filter/easy_map_waypoints_pointcloud_v3.json')
        data = json.load(f)
        a = []
        b = []
        for point in data:
            a.append(point['point_a'])
            b.append(point['point_b'])
        a = np.asarray(a)
        b = np.asarray(b)
        # only use x and y here
        a = a[:, 0:2]
        b = b[:, 0:2]
        b_filtered = []
        for point in b:
            if point[0] < 600 and point[0] > -600 and point[1] < 600 and point[1] > -600:
                b_filtered.append(point)
        b_filtered = np.asarray(b_filtered)
        return a, b_filtered

    def plot_track(self,states):
        a, b = self.__read_map()

        fig, ax = self.__plot_path(states)

        ax.scatter(a[:, 0], a[:, 1])
        ax.scatter(b[:, 0], b[:, 1])

        ax.set_xlim(states[0, 0] - 15, states[0, 0] + 15)
        ax.set_ylim(states[0, 1] - 45, states[0, 1] + 45)

        plt.show()

    def __print_state(self,x):
        print('position x : ' + str(x[0]))
        print('position y : ' + str(x[1]))
        print('velocity x : ' + str(x[2]))
        print('velocity y : ' + str(x[3]))
        v = np.sqrt(x[2]**2+x[3]**2)
        print('combined velocity : ' + str(v))
        print('----------------------------------')

    def __plot_path(self,states):
        fig,ax = plt.subplots(1)
        pos_x = states[0:len(states),0]
        pos_y = states[0:len(states),1]
        plt.scatter(pos_x,pos_y)
        plt.scatter(states[0,0],states[0,1],c='r')
        vehicle_width = 5
        vehicle_length = 10
        vehicle_diagonal = np.sqrt(vehicle_length**2+vehicle_width**2)/2
        alpha = np.arctan(vehicle_width/vehicle_length)
        state_init = states[0]
        angle_init = np.pi/2 - state_init[3]
        beta_init = np.pi/2 - alpha - angle_init
        xy_init = (state_init[0]-np.cos(beta_init)*vehicle_diagonal,state_init[1]-np.sin(beta_init)*vehicle_diagonal)
        rect_init = patches.Rectangle(xy_init
                                      ,vehicle_width,vehicle_length
                                      ,(state_init[3]-np.pi/2)*360/(np.pi*2),linewidth=1
                                      ,edgecolor='r',facecolor='none')
        state_final = states[len(states)-1]
        angle_final = np.pi/2 - state_final[3]
        beta_final = np.pi/2 - alpha - angle_final
        xy_final = (state_final[0]-np.cos(beta_final)*vehicle_diagonal,state_final[1]-np.sin(beta_final)*vehicle_diagonal)
        rect_final = patches.Rectangle(xy_final
                                        ,vehicle_width,vehicle_length
                                        ,(state_final[3]-np.pi/2)*360/(np.pi*2),linewidth=1
                                        ,edgecolor='r',facecolor='none')
        ax.add_patch(rect_init)
        ax.add_patch(rect_final)
        plt.axis('equal')
        return fig,ax
    


    
    
    
    
    
    
    