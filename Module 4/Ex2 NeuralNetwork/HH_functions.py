import numpy as np
import math
import time
import matplotlib.pyplot as plt

def alphaM(V, expm1 = True, print_arg = False):
    if print_arg == 'alphaM':
        print('alphaM:', V, 2.5-0.1*(V+65), np.expm1(2.5 - 0.1*(V + 65)))
    return (2.5 - 0.1*(V+65)) / (np.exp(2.5-0.1*(V + 65)) - 1) if not expm1 else (2.5 - 0.1*(V + 65)) / (np.expm1(2.5 - 0.1*(V + 65)))
              
def betaM(V, expm1 = True, print_arg = False):
    if print_arg == 'betaM':
        print('betaM:', -(V+65)/18)
    return 4*np.exp(-(V+65)/18) 

def alphaH(V, expm1 = True, print_arg = False):
    if print_arg == 'alphaH':
        print('alphaH:', -(V+65)/20)
    return 0.07*np.exp(-(V+65)/20)

def betaH(V, expm1 = True, print_arg = False):
    if print_arg == 'betaH':
        print('betaH:', -(V+65)/18)
    return 1/(np.exp(3.0-0.1*(V+65))+1)

def alphaN(V, expm1 = True, print_arg = False):
    if print_arg == 'alphaN':
        print('betaM:', -(V+65)/18)
    return (0.1-0.01*(V+65)) / (np.exp(1-0.1*(V+65)) - 1) if expm1 else (0.1 - 0.01*(V+65)) / (np.expm1(1-0.1*(V+65)))

def betaN(V, expm1 = True, print_arg = False):
    if print_arg == 'betaN':
        print('betaM:', -(V+65)/80)
    return 0.125*np.exp(-(V+65)/80)

def HH(I_input, N, dt, print_sim_time = False, plot_input = False, expm1 = True, print_arg = False, retrun_gvs = False):
    start_time = time.time()
    T  = N  
    gNa0 = 120.0               # [mS/cm^2]
    ENa  = 115.0               # [mV]
    gK0  = 36.0                # [mS/cm^2]
    EK   = -12.0               # [mV]
    gL0  = 0.3                 # [mS/cm^2]
    EL   = 10.6                # [mV]

    t = np.arange(T, dtype = np.float64) * dt # ms
    V = np.zeros(T, dtype=np.float64)         # mV
    m = np.zeros(T, dtype=np.float64)         
    h = np.zeros(T, dtype=np.float64)
    n = np.zeros(T, dtype=np.float64)

    I0 = I_input(t)
    V[0] = -70.0
    m[0] = 0.05
    h[0] = 0.54
    n[0] = 0.34

    if plot_input:
        plt.figure()
        plt.plot(t, I0)
        plt.ylabel('Input Current (uA)')
        plt.xlabel('Time (ms)')
        plt.show()

    for i in range(0, T-1):
        V[i+1] = V[i] + dt*(gNa0*np.power(m[i], 3)*h[i]*(ENa-(V[i]+65)) + gK0*np.power(n[i], 4)*(EK-(V[i]+65)) + gL0*(EL-(V[i]+65)) + I0[i])
        m[i+1] = m[i] + dt*(alphaM(V[i], expm1 = expm1, print_arg = print_arg)*(1-m[i]) - betaM(V[i], expm1 = expm1, print_arg = print_arg)*m[i])
        h[i+1] = h[i] + dt*(alphaH(V[i], expm1 = expm1, print_arg = print_arg)*(1-h[i]) - betaH(V[i], expm1 = expm1, print_arg = print_arg)*h[i])
        n[i+1] = n[i] + dt*(alphaN(V[i], expm1 = expm1, print_arg = print_arg)*(1-n[i]) - betaN(V[i], expm1 = expm1, print_arg = print_arg)*n[i])
        
    end_time = time.time()
    sim_time = end_time - start_time
    
    if print_sim_time:
        print(f'Simulation Time: {sim_time*1e3:.3f} ms')

    if retrun_gvs:
        return t, V, n, m, h
    return t, V
