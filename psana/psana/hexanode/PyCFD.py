#xiangli@slac.stanford.edu 02/20/2018


import numpy as np
from scipy.optimize import bisect


class PyCFD:

    def __init__(self, params):
        self.sample_interval = params['sample_interval']
        self.delay = int(params['delay']/self.sample_interval)
        self.fraction = params['fraction']
        self.threshold = params['threshold']
        self.walk = params['walk']
        self.polarity = 1 if params['polarity']=='Positive' else -1
        self.timerange_low = params['timerange_low']
        self.timerange_high = params['timerange_high']
        self.offset = params['offset']

        
    def NewtonPolynomial3(self,x,x_arr,y_arr):
    
        d_0_1 = (y_arr[1] - y_arr[0])/(x_arr[1] - x_arr[0])
        d_1_2 = (y_arr[2] - y_arr[1])/(x_arr[2] - x_arr[1])
        d_2_3 = (y_arr[3] - y_arr[2])/(x_arr[3] - x_arr[2])
        
        d_0_1_2 = (d_1_2 - d_0_1)/(x_arr[2] - x_arr[0])
        d_1_2_3 = (d_2_3 - d_1_2)/(x_arr[3] - x_arr[1])        
        d_0_1_2_3 = (d_1_2_3 - d_0_1_2)/(x_arr[3] - x_arr[0])
        
        c0 = y_arr[0]
        c1 = d_0_1
        c2 = d_0_1_2
        c3 = d_0_1_2_3
        
        return c0 + c1*(x-x_arr[0]) + c2*(x-x_arr[0])*(x-x_arr[1]) + c3*(x-x_arr[0])*(x-x_arr[1])*(x-x_arr[2])
        
            
    def CFD(self,wf, wt):        
        
        wf = wf[(wt>self.timerange_low)&(wt<self.timerange_high)] 
        wt = wt[(wt>self.timerange_low)&(wt<self.timerange_high)] #choose the time window of interest        
        
        wf_1 = wf[:-self.delay] #original waveform
        wf_2 = wf[self.delay:] #delayed waveform
       
        wf_cal = wf_1 - self.fraction*wf_2 #bipolar waveform
        wf_cal_m_walk = self.polarity*wf_cal-self.walk+self.polarity*(self.fraction*self.offset-self.offset) #bipolar signal minus the walk level
        wf_cal_m_walk_sign = np.sign(wf_cal_m_walk) 

        wf_cal_ind = np.where((wf_cal_m_walk_sign[:-1] < wf_cal_m_walk_sign[1:]) & 
        (wf_cal_m_walk_sign[1:] != 0) & ((wf_cal_m_walk[1:] - wf_cal_m_walk[:-1]) >= 1e-8))[0] #find the sign change locations of wf_cal_m_walk

        #check if the orignal signal is above the threhold at sign change locations of wf_cal_m_walk
        wf_cal_ind_ind = np.where(self.polarity*wf_1[wf_cal_ind] > (self.threshold+self.polarity*self.offset))[0]  

        
        t_cfd_list = []
        
        
        #The arrival time t_cfd is obtained from the Newton Polynomial fitted to the 4 data points around the location found from above.
        for ind in wf_cal_ind_ind:

            t_arr = wt[(wf_cal_ind[ind]-1):(wf_cal_ind[ind]+3)]

            wf_cal_m_walk_arr = wf_cal_m_walk[(wf_cal_ind[ind]-1):(wf_cal_ind[ind]+3)]
            
            if len(t_arr) != 4 or len(wf_cal_m_walk_arr) != 4:
                continue
            
            if (t_arr[1] - t_arr[0])==0 or (t_arr[2] - t_arr[1])==0 or (t_arr[3] - t_arr[2])==0:
                continue
                
            if (t_arr[2] - t_arr[0])==0 or (t_arr[3] - t_arr[1])==0 or (t_arr[3] - t_arr[0])==0:
                continue
            
            t_cfd = bisect(self.NewtonPolynomial3,t_arr[1],t_arr[2],args=(t_arr, wf_cal_m_walk_arr),xtol=1e-3)
            
            t_cfd_list.append(t_cfd)

        return t_cfd_list
