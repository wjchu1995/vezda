# Copyright 2017-2019 Aaron C. Prunty
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#        
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#==============================================================================

import numpy as np
from scipy.signal import butter, lfilter, tukey
from vezda.math_utils import nextPow2

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=1)
    return y


def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)


def add_noise(data, dt, min_freq, max_freq, snr=2):
    '''
    data: 3D input array of shape Nr x Nt x Ns
    Nr: number of receivers
    Nt: number of time samples
    Ns: number of sources
    dt: sampling interval in time
    min_freq: minimum frequency component of the generated noise
    max_freq: maximum frequency component of the generated noise
    snr: specified signal-to-noise ratio
    '''
    
    Nr, Nt, Ns = data.shape
    
    # average signal power per recording
    signalPower = np.sum(data**2, axis=(0, 1)) / Nr
    noisyData = np.zeros((Nr, Nt, Ns), dtype=data.dtype)
    for r in range(Nr):
        for s in range(Ns):
            noise = band_limited_noise(min_freq, max_freq, Nt, 1 / dt)
            noisePower = np.sum(noise**2)
            scale = np.sqrt(signalPower[s] / (noisePower * snr))
            noisyData[r, :, s] = data[r, :, s] + scale * noise
    
    return noisyData


def compute_spectrum(data, dt, power=False):
    
    Nr, Nt, Ns = data.shape
    N = nextPow2(Nt)
    freqs = np.fft.rfftfreq(N, dt)
    amplitudes = np.abs(np.fft.rfft(data, axis=1, n=N))
    if power:
        amplitudes = amplitudes**2 / N
    
    A = np.sum(amplitudes, axis=(0, 2)) / (Nr * Ns)
    
    return freqs, A


def tukey_taper(X, dt, peakFreq):
    Nt = X.shape[1]
    # Np : Number of samples in the dominant period T = 1 / peakFreq
    Np = int(round(1 / (dt * peakFreq)))
    # alpha is set to taper over 6 of the dominant period of the
    # pulse function (3 periods from each end of the signal)
    alpha = 6 * Np / Nt
    print('Tapering time signals with Tukey window: %d'
          %(int(round(alpha * 100))) + '%')
    TukeyWindow = tukey(Nt, alpha)
    X *= TukeyWindow[None, :, None]
    
    return X
