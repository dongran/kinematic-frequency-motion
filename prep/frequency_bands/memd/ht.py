#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:41:24 2020

@author: dong
"""

import math

import numpy as np


def FAhilbert(imfs, dt):
    """
    Performs Hilbert transformation on imfs.
    Returns frequency and amplitude of signal.
    """
    n_imfs = imfs.shape[0]
    f = []
    a = []
    for i in range(n_imfs - 1):
        inst_imf = imfs[i, :]
        inst_amp, phase = hilb(inst_imf, unwrap=True)
        inst_freq = (1 / dt) * np.diff(phase) / (2 * math.pi)

        if (imfs[i, :] == 0).all():
            inst_freq = np.zeros([len(inst_freq)])
            inst_amp = np.zeros([len(inst_amp)])

        f.append(inst_freq)
        a.append(inst_amp)
    return np.asarray(f).T, np.asarray(a).T


def FAhilbert2(imfs, dt):
    """
    Performs Hilbert transformation on imfs.
    Returns frequency and amplitude of signal.
    """
    n_imfs = imfs.shape[0]
    f = []
    a = []
    for i in range(n_imfs):
        inst_imf = imfs[i, :]
        inst_amp, phase = hilb(inst_imf, unwrap=True)
        inst_freq = (1 / dt) * np.diff(phase) / (2 * math.pi)

        if (imfs[i, :] == 0).all():
            inst_freq = np.zeros([len(inst_freq)])
            inst_amp = np.zeros([len(inst_amp)])

        f.append(inst_freq)
        a.append(inst_amp)
    return np.asarray(f).T, np.asarray(a).T


def hilb(s, unwrap=False):
    """
    Performs Hilbert transformation on signal s.
    Returns amplitude and phase of signal.
    Depending on unwrap value phase can be either
    in range [-pi, pi) (unwrap=False) or
    continuous (unwrap=True).
    """
    from scipy.signal import hilbert

    H = hilbert(s)
    amp = np.abs(H)
    phase = np.arctan2(H.imag, H.real)
    if unwrap:
        phase = np.unwrap(phase)
    return amp, phase

