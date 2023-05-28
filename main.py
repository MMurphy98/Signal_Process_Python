# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import scipy.io as scio
import scipy.signal
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

def My_FFT(x, Fs, win):
    N = len(x)
    index_tail = int(N/2) + 1
    x_win = x * win
    xft = fft(x_win, N)
    xft = xft[0:index_tail]
    xpsd = abs(xft)**2 * (1/(Fs*N))
    xpsd[1:-2] = 2*xpsd[1:-2]
    freq = np.arange(0, N/2+1) * Fs/N
    return freq, xpsd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fs = 1000
    fin = 101
    t = np.arange(0,2000) / fs
    x = 0.85*np.cos(2*np.pi*fin*t) + 0.05*np.cos(2*np.pi*2*fin*t)
    N = len(x)
    pff, pxx = My_FFT(x, fs, np.ones(N))

    plt.figure()
    plt.plot(pff, 10*np.log10(pxx))
    plt.xlim([0, fs/2])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power/Frequency [dB]")
    plt.title("Periodogram Using FFT")
    plt.grid()
    plt.show()




    

