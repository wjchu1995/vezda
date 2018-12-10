import numpy as np
from vezda.math_utils import nextPow2

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
    
    signalPower = np.sum(data**2, axis=1)
    Nr, Nt, Ns = data.shape
    noisyData = np.zeros((Nr, Nt, Ns), dtype=data.dtype)
    for r in range(Nr):
        for s in range(Ns):
            noise = band_limited_noise(min_freq, max_freq, Nt, 1 / dt)
            noisePower = np.sum(noise**2)
            scale = np.sqrt(signalPower[r, s] / (noisePower * snr))
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
