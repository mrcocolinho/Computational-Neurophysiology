import time
import numpy as np
from scipy.integrate import odeint

class HH_simulator:

    def __init__(self, params = {'gNa0': 120.0, 'ENa' : 115.0, 'gK0': 36.0,'EK': -12.0,'gL0': 0.3,'EL': 10.6, 'Cm': 1.0}):

        # unpack the physiological parameters
        self.gNa0 = params['gNa0']    # mS/cm^2
        self.ENa = params['ENa']      # mV
        self.gK0 = params['gK0']      # mS/cm^2
        self.EK = params['EK']        # mV
        self.gL0 = params['gL0']      # mS/cm^2
        self.EL = params['EL']        # mV
        self.Cm = 1.0                 # uF/cm^2

        # initial conditions
        self.V0 = -70
        self.n0 = 0.34
        self.m0 = 0.05
        self.h0 = 0.54

    def alphaM(self, V, expm1 = True):
        return (2.5 - 0.1*(V+65.0)) / (np.exp(2.5-0.1*(V + 65.0)) - 1.0) if not expm1 else (2.5 - 0.1*(V + 65.0)) / (np.expm1(2.5 - 0.1*(V + 65.0)))
              
    def betaM(self, V):
        return 4*np.exp(-(V+65.0)/18.0) 
    
    def alphaH(self, V):
        return 0.07*np.exp(-(V+65.0)/20.0)
    
    def betaH(self, V):
        return 1/(np.exp(3.0-0.1*(V+65.0))+1.0)
    
    def alphaN(self, V, expm1 = True):
        return (0.1-0.01*(V+65.0)) / (np.exp(1.0-0.1*(V+65.0)) - 1.0) if expm1 else (0.1 - 0.01*(V+6.0)) / (np.expm1(1.0-0.1*(V+65.0)))
    
    def betaN(self, V):
        return 0.125*np.exp(-(V+65)/80)

    def dHH(self, y, t):

        # get output vector
        dy = np.zeros((4,))

        # unpack the vector
        V, n, m, h = [y[i] for i in range(4)]
        
        # derivatives
        dy[0] = self.gNa0*np.power(m, 3)*h*(self.ENa - (V + 65.0)) + self.gK0*np.power(n, 4)*(self.EK-(V+65.0)) + self.gL0*(self.EL-(V+65.0)) + self.I(t)
        dy[1] = self.alphaN(V)*(1.0 - n) - self.betaN(V)*n
        dy[2] = self.alphaM(V)*(1.0 - m) - self.betaM(V)*m
        dy[3] = self.alphaH(V)*(1.0 - h) - self.betaH(V)*h
        
        return dy

    def simulate(self, I_input, N, fs, print_sim_time = False, return_gvs = False):
        '''
        I_input : Input current function (uA)
        fs      : sample frequency (kHz)
        N       : length of input
        '''
        start_time = time.time()

        # input current
        self.I = I_input
        
        # set the initial conditons
        y0 = np.array([self.V0, self.n0, self.m0, self.h0])

        # get the time vector (ms)
        self.t = np.arange(N) / fs

        # get the output
        y = odeint(self.dHH, y0, self.t)

        # unpack
        self.V, self.n, self.m, self.h = [y[:,i] for i in range(4)]
        
        end_time = time.time()
        self.sim_time = end_time - start_time
        if print_sim_time:
            print(f'Simulation Time: {self.sim_time*1e3:.3f} ms')
        
        if return_gvs: 
            return t, self.V, self.n, self.m, self.h
        return self.t, self.V