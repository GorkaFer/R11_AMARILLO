#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:35:35 2022

@author: aduo
"""

import sys
from contextlib import closing
from io import StringIO
from typing import Optional
import numpy as np
from gymnasium import Env, spaces
import pandas as pd
import matplotlib.pyplot as plt
import time
#plt.switch_backend('QT5Agg')
#plt.style.use("ggplot")
class motorEnv(Env):
    
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self):
        super(motorEnv, self).__init__()
        # Cargar datos del entorno
        # print("open 3")
        self.rot = pd.read_csv('Datos_v1.csv')
        self.rot = self.rot.drop('index', axis=1)
        self.rot['col'] = '0'
        # print(self.rot)
        self.num_states = len(self.rot)
        num_rows = len(self.rot)
        num_columns = self.rot.shape[1]
        max_row = num_rows - 1
        max_col = num_columns - 1
        self.new_state = None
        self.old_state = None
        
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf]),
                                            high=np.array([np.inf, np.inf, np.inf]),dtype=np.float32)
        
        

        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.line1 = self.ax.scatter(self.rot['var1'], self.rot['var2'],c=self.rot['col'])
        

    def step(self, a):
        done=False
        if a == 0:
            var1_prima = self.var1 + 1
            try:
                self.new_state = self.rot[(self.rot['var1'] == var1_prima) &
                                          (self.rot['var2'] == self.var2)].index[0]
                self.var1=var1_prima
            except:
                self.new_state = self.old_state
                reward = -100

        elif a == 1:
            var1_prima = self.var1 - 1
            try:
                self.new_state = self.rot[(self.rot['var1'] == var1_prima) &
                                          (self.rot['var2'] == self.var2)].index[0]
                self.var1=var1_prima
            except:
                self.new_state = self.old_state
                reward = -100
        elif a == 2:
            var2_prima = self.var2 + 1
            try:
                self.new_state = self.rot[(self.rot['var1'] == self.var1) &
                                          (self.rot['var2'] == var2_prima)].index[0]
                self.var2=var2_prima
            except:
                self.new_state = self.old_state
                reward = -100
        elif a == 3:
            var2_prima = self.var2 - 1
            try:
                self.new_state = self.rot[(self.rot['var1'] == self.var1) &
                                          (self.rot['var2'] == var2_prima)].index[0]
                self.var2 = var2_prima
            except:
                self.new_state = self.old_state
                reward = -100

        if (self.rot.iloc[self.new_state, 2] == self.rot['w'].min()):
            reward = 1000
            done = True
            print("reward: 1000")
            # self.figure, self.ax = plt.subplots(figsize=(10, 8))
            # self.line1 = self.ax.scatter(self.rot['var1'], self.rot['var2'],c=self.rot['col'])
        elif self.new_state != self.old_state:
            reward = -1
            done = False
            
        self.rot.iloc[self.new_state, 3] = '1'
        # self.ax.plot(self.var1,self.var2,'o')
        self.old_state = self.new_state
        return self.new_state, reward, done, {}

    def reset(self):
        self.var1 = self.rot.iloc[0, 0]
        self.var2 = self.rot.iloc[0, 1]
        # print(self.var1)
        # print(self.var2)
        self.old_state = 0
        self.rot.iloc[0, 3] = '1'
        #self.render()
        return self.old_state

    def render(self):
        #print(self.rot)
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.scatter(self.rot['var1'], self.rot['var2'], c=self.rot['col'])
        self.line1.set_color(self.rot['col'])
        #self.figure.canvas.draw()
        #self.figure.canvas.flush_events()
        #time.sleep(0.1)
        return 0