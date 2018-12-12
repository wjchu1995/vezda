import numpy as np

# These parameters must be defined in pulseFun.py
# and named as follows (units can be different)

velocity = 2      # km/s
peakFreq = 25     # peak frequency (Hz)
peakTime = 0.1 	  # seconds

# definition of the pulse function
def pulse(t):
    return (1.0 - 2.0 * (np.pi**2) * (peakFreq**2) * (t - peakTime)**2) * np.exp(-(np.pi**2) * (peakFreq**2) * (t - peakTime)**2)
