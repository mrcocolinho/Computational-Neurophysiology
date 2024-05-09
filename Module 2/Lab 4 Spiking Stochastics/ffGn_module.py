"""SV: This implementation is based on the oryginal MATLAB implementation of zilany2009, and comes from the  implementation of the python package cochlea-master-repo written by mrkrd/cochlea on github, remains to be tested whether this is idem to the original zilany/Scott/Jackson, and whether it is equivalent to the other python package for FFGN SV found online, and Hari's ffGN file online.

TODO: Unit test against the oryginal. => SV: comment from authors of this file!?

"""

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.random import randn
from scipy.signal import resample
from numpy.fft import fft, ifft
import random

# sigma has been removed as it is never read
def ffGn(N, tdres, Hinput, mu):

    assert (N > 0)
    assert (tdres < 1)
    assert (Hinput >= 0) and (Hinput <= 2)


    # Downsampling No. of points to match with those of Scott jackson (tau 1e-1) => SV: makes no sense, leave out for now
    #resamp = int(np.ceil(1e-1 / tdres))
    #nop = N
    #N = int(np.ceil(N / resamp) + 1)
    #if N < 10:
    #    N = 10

    # Determine whether fGn or fBn should be produced.
    if Hinput <= 1:
        H = Hinput
        fBn = 0
    else:
        H = Hinput - 1
        fBn = 1


    # Calculate the fGn.
    if H == 0.5:
        # If H=0.5, then fGn is equivalent to white Gaussian noise.
        #y = randn(N) #orignal function
        #SV added the standard deviations here, was always sigma = 1 in original 
        if mu < 1.1:
            sigma = 1
        elif mu < 18:
            sigma = np.sqrt(mu*9)        # we use the same for MS than HS
        else:
            sigma = np.sqrt(mu*9)        # we use the same for MS than HS  
            
        y = np.random.normal(0,sigma,N) #SV
        return y
        
    else:
        # TODO: make variables persistant
        Nfft = int(2 ** np.ceil(np.log2(2*(N-1))))
        NfftHalf = np.round(Nfft / 2)

        k = np.concatenate( (np.arange(0,NfftHalf), np.arange(NfftHalf,0,-1)) )
        Zmag = 0.5 * ( (k+1)**(2*H) -2*k**(2*H) + np.abs(k-1)**(2*H) )

        Zmag = np.real(fft(Zmag))
        assert np.all(Zmag >= 0)

        Zmag = np.sqrt(Zmag)

        Z = Zmag * (randn(Nfft) + 1j*randn(Nfft))

        y = np.real(ifft(Z)) * np.sqrt(Nfft)

        y = y[0:N]

        # Convert the fGn to fBn, if necessary.
        if fBn == 1:
            y = np.cumsum(y)


        # Resampling to match with the input resolution => SV questions whether this makes sense..
        #y = resample(y, resamp*len(y))

     #THESE ARE ORIGINAL ZILANY 2009 values, we shall adjust these to our needs.
     # mu is the spontaneous rate of the fiber LS, MS and LS
     #   if mu < 0.5:
     #       sigma = 5
     #   elif mu < 18:
     #       sigma = 50          # 7 when added after powerlaw
     #   else:
     #       sigma = 200         # 40 when added after powerlaw
        
        
     #THESE ARE SV FITS BASED ON OTHER AN MODEL   
        if mu < 1.1:
            sigma = 1
        elif mu < 18:
            sigma = np.sqrt(mu*9)        # we use the same for MS than HS
        else:
            sigma = np.sqrt(mu*9)        # we use the same for MS than HS  


        y = y*sigma

        #return y[0:nop]
        return y

    
def inhomPP(Rate, dt):

    indx = 0
    inhPS = []
    eis = []
    VarA = 0
    exitB = False
    outi = 0
    Stime = 0
    
    while indx < len(Rate):  
        t = 0
        VarA = 0
        ei=random.expovariate(1) #/dt  
        #spike times, which are generate at an average rate of 1 spike/s. 
        #output is in t until the next spike: t = samples * dt #so t/dt = samples (ei unit).
        #print(ei)
        exitA = False
        while not exitA and indx + 1 < len(Rate):
            indx += 1
            #print(indx)
            VarA += Rate[indx]*dt #in [Rate=samples/s * dt] 
            #print(VarA * dt)
            t += dt  #is the minimum time for which ti+1 > ei
            exitA = (VarA>ei) #will generate a true when condition is met   
        if exitA:
            Stime += t
            #print(Stime)
            #print(indx)
            inhPS.append(Stime)
            eis.append(ei)
        else:
            break
            #inhPS[outi] = Stime # can only do when knowing the size in advance
        
    inhPS = np.array(inhPS)
    return inhPS, eis
    

def main():
    y = ffGn(10, 1e-1, 0.2, 1, debug=True)
    print(y)


if __name__ == "__main__":
    main()
