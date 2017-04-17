# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:18:50 2017

@author: johns
"""
import numpy as np
import pandas as pd 
import scipy.optimize as opt

class SpringTrial: 
    
    fitfunc = lambda p, t: p[0] * np.cos((p[1] * t) + p[2]) + p[3]
    errfunc = lambda p, t, y: SpringTrial.fitfunc(p, t) - y

    
    AMPLITUDE = 0 
    OMEGA = 1 
    PHI = 2 
    Y_0 = 3
    
    def __init__(self, m, i): 
        self.mass = m         
        self.trial = i
        self.filename = self.get_filename()
        self.data = pd.read_csv(self.filename) 
        self.data_filtered = self.data[self.data["Position"] > 0.19]
        self.p0 = list(self.get_p0())
        self.p1, success = opt.leastsq(SpringTrial.errfunc, self.p0, (self.data_filtered["Time"], self.data_filtered["Position"]))        
        self.k = (self.p1[1] ** 2) * (self.mass + 0.00123)
        
    def get_p0(self): 
        mean = self.data["Position"].describe()["mean"] 
        delta_y = (self.data["Position"].describe()["max"] - self.data["Position"].describe()["min"])
        amp = delta_y / 2
        peaks = self.get_peaks() 
        if(len(peaks) > 2):
            peak_sum = 0
            for i in range(len(peaks) - 1): 
                peak_sum += self.data["Time"][peaks[i + 1]] - self.data["Time"][peaks[i]]
            avg_pd = peak_sum / (len(peaks) - 1) 
        else: 
            avg_pd = self.data["Time"].max() / 2
        phase = (2 * np.pi / avg_pd) * peaks[0] 
        return amp, (2 * np.pi / avg_pd), phase, mean
        
    def get_peaks(self): 
        peaks = list()
        avg = self.data["Position"].mean() 
        for i in range(len(self.data) - 2): 
            if(self.data["Position"][i+1] > avg):
                cut = self.data["Position"][i : i + 2 + 1]
                if((cut[i] <= cut[i + 1]) and (cut[i + 1] > cut[i + 2])): 
                    peaks.append(i + 1)
        return peaks
        
    def get_filename(self): 
        if(self.mass < 1): 
            mass_str = str(int(self.mass * 1000)) + "g"
            
        else: 
            mass_str = "{:.1f}kg".format(self.mass) 
        return "data/Mass{}Trial{}.csv".format(mass_str, self.trial)