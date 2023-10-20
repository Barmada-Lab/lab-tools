from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage # type: ignore
import math

def freq_norm_sq(shape):
    i_half = np.pi * np.linspace(0, 1, shape[0] // 2)
    j_half = np.pi * np.linspace(0, 1, shape[1] // 2)

    i = np.concatenate((i_half, i_half[::-1]))
    j = np.concatenate((j_half, j_half[::-1]))

    I, J = np.meshgrid(i, j, indexing ='ij')

    return I**2 + J**2

def hdaf_lowpass(shape: Tuple[int, int], radius: int, n=60):
    """Fourier space"""
    cutoff_freq = 0.542423208/radius + 0.0957539421/radius**2
    cnk = np.sqrt(2*n + 1) / (cutoff_freq*np.sqrt(2)*np.pi)

    coefficients = [1 / math.factorial(i) for i in range(n)][::-1]
    x_i = freq_norm_sq(shape) * cnk**2

    filt = np.polyval(coefficients, x_i) * np.exp(-x_i)
    return filt

def hdaf_highpass(shape: Tuple[int, int], radius: int, n=60):
    """Fourier space"""
    return 1 - hdaf_lowpass(shape, radius, n)

def hdaf_bandpass(shape: Tuple[int, int], radii: Tuple[int,int], n=60):
    """Fourier space"""
    r0, r1 = radii
    bandpass = hdaf_lowpass(shape, r0, n) - hdaf_lowpass(shape, r1, n)
    return -bandpass if r0 > r1 else bandpass

def hdaf_laplacian(shape: Tuple[int, int], radius: int, n=60):
    """Fourier space"""
    return -freq_norm_sq(shape) * hdaf_lowpass(shape, radius, n)

def oriented_gaussian(theta: float, sigma: Tuple[float, float], sigma_truncate: float = 4.0):
    """
    Anisotropic, steerable 2d gaussian (you can shape and rotate it)

    params:
    - theta (float): angle of orientation in radians (range [0, pi])
    - sigma (float): sigmas for gaussian along the primary and secondary axis
    - sigma_truncate (float): num of sigmas to truncate at
    """

    sigma_alpha, sigma_beta = sigma
    width = int(sigma_truncate * (sigma_alpha * abs(np.cos(theta)) + sigma_beta * np.sin(theta)))
    height = int(sigma_truncate * (sigma_alpha * np.sin(theta) + sigma_beta * abs(np.cos(theta))))

    x_lim = max(width, 10)
    y_lim = max(height, 10)
    xv, yv = np.meshgrid(
        np.linspace(-x_lim, x_lim, 2 * x_lim),
        np.linspace(-y_lim, y_lim, 2 * y_lim), indexing='ij')

    norm = 1 / (2 * np.pi * sigma_alpha * sigma_beta)
    x_trans = xv * np.cos(theta) + yv * np.sin(theta)
    y_trans = -xv * np.sin(theta) + yv * np.cos(theta)
    gauss = norm * np.exp((-1/2) * (x_trans**2 / (2*sigma_alpha**2) + y_trans**2 / (2*sigma_beta**2)))
    return gauss

def directional_ratio(img: ArrayLike, sigma: Tuple[int,int], n: int = 8):
    filtered = []
    for i in range(n):
        theta = i * np.pi / n
        gaussian = oriented_gaussian(theta, sigma)
        filtered.append(ndimage.convolve(img, gaussian))

    filtered = np.array(filtered)
    mins = filtered.min(axis=0)
    maxs = filtered.max(axis=0)
    return mins / maxs + 1e-6
