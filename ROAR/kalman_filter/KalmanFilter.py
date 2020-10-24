# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:12:46 2020

@author: Ethan
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from filterpy.kalman import ExtendedKalmanFilter

class Kalman_Filter:
    # state variables (x_position,y_position,x_velocity,y_velocity)
    current_state : np.array((4,)) 
    # state inputs (x_acceleration,y_acceleration (refers to steering & throttle))
    system_inputs : np.array((2,)) 
    predicted_state_current : np.array((4,))
    delta_T : float
    SYSTEM_MATRIX : np.array((4,4))     #State Transition Matrix
    INPUT_MATRIX : np.array((4,2))

    
    def __init__(self,initial_state,delta_T):
        self.current_state = initial_state
        self.predicted_state_current = self.current_state
        self.delta_T = delta_T
        self.SYSTEM_MATRIX = np.array([[1,0,delta_T,0],[0,1,0,delta_T],[0,0,1,0],[0,0,0,1]])
        self.INPUT_MATRIX = np.array([[0.5*delta_T**2,0],[0,0.5*delta_T**2],[delta_T,0],[0,delta_T]])
        print('SYSTEM MATRIX : ')
        print(self.SYSTEM_MATRIX)
        print('INPUT MATRIX : ')
        print(self.INPUT_MATRIX)
        
    def predict(self,state,inputs_xy):
        v_x = state[2]
        v_y = state[3]
        theta = np.arctan2(v_x,v_y)
        # print('theta : '+str(theta*360/(2*np.pi)))
        inputs = self.__carfixed_to_xy(inputs_xy,theta)
        predicted_next_state = self.SYSTEM_MATRIX@state + self.INPUT_MATRIX@inputs
        self.__print_state(predicted_next_state)
        self.predicted_state_current = predicted_next_state
        return self.predicted_state_current
    
    def update(self,measurements,inputs_xy):
        predicted_state = self.predict(self.current_state, inputs_xy)
        self.current_state = predicted_state
        if measurements != None:
            self.current_state[0] = measurements[0]
            self.current_state[1] = measurements[1]
        
    def predict_plot_future(self,prediction_time, plot=False):
        self.predicted_state_current = self.current_state
        states=[]
        for i in range(1,int(prediction_time/self.delta_T)):
            next_state = self.predict(self.predicted_state_current,self.system_inputs)
            states.append(next_state)
        states = np.asarray(states)
        if plot:    
            fig,ax= self.plot_path(states)
        return states

    def __carfixed_to_xy(self,vector_carfixed,theta):
        vector_xy = np.zeros(2,)
        vector_xy[0] = vector_carfixed[0]*np.cos(theta)+vector_carfixed[1]*np.sin(theta)
        vector_xy[1] = -vector_carfixed[0]*np.sin(theta)+vector_carfixed[1]*np.cos(theta)
        # print(vector_xy)
        return vector_xy

    def __print_state(self,x):
        print('position x : ' + str(x[0]))
        print('position y : ' + str(x[1]))
        print('velocity x : ' + str(x[2]))
        print('velocity y : ' + str(x[3]))
        v = np.sqrt(x[2]**2+x[3]**2)
        print('combined velocity : ' + str(v))
        print('----------------------------------')

    def plot_path(self,states):
        fig,ax = plt.subplots(1)
        pos_x = states[0:len(states),0]
        pos_y = states[0:len(states),1]
        plt.scatter(pos_x,pos_y)
        plt.scatter(states[0,0],states[0,1],c='r')
        vehicle_width = 5
        vehicle_length = 10
        vehicle_diagonal = np.sqrt(vehicle_length**2+vehicle_width**2)/2
        alpha = np.arctan(vehicle_width/vehicle_length)
        angle_init = np.arctan2(states[0,2],states[0,3])
        beta_init = np.pi/2 - alpha - angle_init
        xy_init = (states[0,0]-np.cos(beta_init)*vehicle_diagonal,states[0,1]-np.sin(beta_init)*vehicle_diagonal)
        rect_init = patches.Rectangle(xy_init
                                      ,vehicle_width,vehicle_length
                                      ,-angle_init*360/(np.pi*2),linewidth=1
                                      ,edgecolor='r',facecolor='none')
        angle_final = np.arctan2(states[len(states)-1,2],states[len(states)-1,3])
        beta_final = np.pi/2 - alpha - angle_final
        xy_final = (states[len(states)-1,0]-np.cos(beta_final)*vehicle_diagonal,states[len(states)-1,1]-np.sin(beta_final)*vehicle_diagonal)        
        rect_final = patches.Rectangle(xy_final
                                        ,vehicle_width,vehicle_length
                                        ,-angle_final*360/(np.pi*2),linewidth=1
                                        ,edgecolor='r',facecolor='none')
        ax.add_patch(rect_init)
        ax.add_patch(rect_final)
        plt.axis('equal')
        return fig,ax
    


    
    
    
    
    
    
    