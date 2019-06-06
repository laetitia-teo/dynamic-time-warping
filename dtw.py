import os.path as op
import numpy as np
import wave
import matplotlib.pyplot as plt

from numpy.fft import fft
from scipy.fftpack import dct

from tqdm import tqdm
from glob import glob

### =========================== Utility functions =============================

def wav_to_signal(path):
    """
    Transforms a wav file into a numerical sequence. 
    """
    with wave.open(path, 'rb') as f:
        signal = f.readframes(-1)
        return np.fromstring(signal, 'Int16')

def full_tri(n):
    """
    Returns a triangle.
    Array of length n whose value form a triangle centered at half the array
    with value 1, and is zero-valued at the edges.
    """
    a = np.arange(n)
    a = np.minimum(a, np.flip(a))
    a = a / np.max(a)
    return a

def tri(n, a, b):
    """
    Returns a vector of length n, identically equal to 0 except beween a and
    b where there is a symmetric triangle centered at (a + b)/2
    """
    assert((a < b) and (0 <= a) and (b <= n))
    zeros_before = np.zeros(a)
    triangle = full_tri(b - a)
    zeros_after = np.zeros(n - b)

    return np.concatenate((zeros_before, triangle, zeros_after))

def linear_scale(n, c):
    return ((np.arange(c + 2) / (c + 2) * n)).astype(int)

def tri_matrix(n, c):
    """
    Defines a family (arranged as a matrix) of triangular filters, with 
    n the length of the filters and c the number of filters.
    """
    small = int(n / (c+2))  # trailing edges at both ends of the filter vector
    n = n + 2*small
    indices = linear_scale(n, c)
    vectorlist = []
    for i in range(c):
        vectorlist.append(tri(n, indices[i], indices[i + 2]))
    return np.array(vectorlist)[:, small:-small].T

### ==================== Computation of the cepstral coeffs ===================

def cepstral(s, c=26, size=256, shift=32, show=False):
    """
    Extracts the ceptral parameters of the signal s.
    """

    length = len(s)
    s_hat = []

    # log of the fourier energy coefficients
    for i in range(0, length - size, shift):
        s_hat.append(np.log(np.abs(fft(s[i:i+size] * full_tri(size)))**2))
    s_hat = np.vstack(s_hat)
    s_hat = s_hat[..., :int(size/2)]

    if show:
        plt.matshow(s_hat.T)
        plt.colorbar()
        plt.show()

    # pass the fourier coefficients through a linear triangular filter
    n = s_hat.shape[1]
    filters = tri_matrix(n, c) # shape (n, c)
    s_hat = s_hat.dot(filters) # shape (length, c)

    # discrete cosinus transform of the modified fourier coefs
    s_tilde = dct(s_hat)

    if show:
        plt.matshow(s_tilde.T)
        plt.colorbar()
        plt.show()

    return s_tilde

### =============================== Dynamic Time Warping ======================

def dtw(x, y, gamma=1, show=False):
    """
    Dynamic Time Warping algorithm.
    """
    n, m = len(x), len(y)
    # distance matrix
    d = x[:, np.newaxis, :] - y[np.newaxis, ...]
    d = np.sum(d**2, axis=-1)**0.5
    # cumulative distance matrix
    c = np.zeros(d.shape)
    c[0, :] = np.cumsum(d[0, :])
    c[:, 0] = np.cumsum(d[:, 0])

    def dynamic_min_path(i, j):
        """
        Sets the value of the cumulative distance matrix and of the 
        coding matrix.
        """
        topleft = c[i -1, j - 1] + gamma * d[i, j]
        top = c[i - 1, j] + d[i, j]
        left = c[i, j - 1] + d[i, j]
        # b[i, j] = np.argmin([topleft, top, left])
        c[i, j] = np.min([topleft, top, left])

    # computing the cumulative distances
    for i in range(1, min(n, m)):
        dynamic_min_path(i, i)  # diagonal term
        for j in range(i, n):  # on the x-axis first
            dynamic_min_path(j, i)
        for k in range(i, m):  # then on the y-axis
            dynamic_min_path(i, k)

    # then getting, from the end to the beginning, the optimal path :
    idx = (n - 1, m - 1)
    # distance = 0.0
    distance = c[idx]
    pstar = [idx]
    normalisation = 0.
    while idx != (0, 0):
        norm = 1.
        if idx[0] == 0:
            idx = (idx[0], idx[1] - 1)
        elif idx[1] == 0:
            idx = (idx[0] - 1, idx[1])
        else:
            idxlist = [(idx[0] - 1, idx[1] - 1), 
                       (idx[0] - 1, idx[1]),
                       (idx[0], idx[1] - 1)]
            valuelist = [c[idx] for idx in idxlist]
            i = np.argmin(valuelist)
            if i == 0:
                norm = gamma
            idx = idxlist[i]
        # distance += c[idx]
        pstar.append(idx)
        normalisation += norm

    if show:
        plt.matshow(d)
        a = np.zeros(c.shape)
        for idx in pstar:
            a[idx] = 1
        plt.matshow(a)
        plt.show()

    return distance/norm, pstar

### ======================= Application to speech data ========================

# first evaluation :

path0 = op.join('Data', 'Data', 'SIG', 'SIG_Rep_1_Number_0.wav')
paths = glob(op.join('Data', 'Data', 'SIG', '*.wav'))

s0 = wav_to_signal(path0)
ss = [wav_to_signal(path) for path in paths]

true = ['Number_0' in path for path in paths]
distances = [dtw(cepstral(s0), cepstral(s))[0] for s in tqdm(ss)]

ds = np.array(distances)
ds = ds / np.max(ds)

true = np.array(true)
 
# we'll take as separation boundary the middle between the highest member of
# same class and the lowest of the other classes.

highest_true = np.max(ds - (1 - true))
lowest_false = np.min(ds + true)
boundary = (highest_true + lowest_false) / 2

pred = np.array(((ds - boundary) <= 0).astype(int))

num_wrong = np.sum(np.abs(true - pred))
acc = np.sum(true  * pred) / np.sum(pred)

# use several
