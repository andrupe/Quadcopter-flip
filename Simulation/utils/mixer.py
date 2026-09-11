# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import math

import numpy as np
import config


def mixerFM(quad, thr, moment):
    """
    Mixer: (collective thrust, body moments) -> per-rotor angular speed commands.

    W = sqrt(clip(M^-1 @ [thr, mx, my, mz], w_min^2, w_max^2))

    Written with scalars rather than np.dot/np.clip/np.sqrt: this runs once per physics
    substep (10x per environment step) and the numpy dispatch on a 4x4 @ 4-vector costs
    more than the arithmetic. Same expression, same order, same clamping semantics and
    same NaN propagation as the vector form - see scratch/check_fast_math.py.
    """
    mixer_f_minv = quad.params["mixerFMinv"]
    thr = float(thr)
    m0 = float(moment[0])
    m1 = float(moment[1])
    m2 = float(moment[2])
    w_lo = float(quad.params["minWmotor"]) ** 2
    w_hi = float(quad.params["maxWmotor"]) ** 2

    w_cmd = np.empty(4, dtype=np.float64)
    for i in range(4):
        row = mixer_f_minv[i]
        v = (float(row[0]) * thr + float(row[1]) * m0
             + float(row[2]) * m1 + float(row[3]) * m2)
        if v < w_lo:
            v = w_lo
        elif v > w_hi:
            v = w_hi
        w_cmd[i] = math.sqrt(v)

    return w_cmd


## Under here is the conventional type of mixer

# def mixer(throttle, pCmd, qCmd, rCmd, quad):
#     maxCmd = quad.params["maxCmd"]
#     minCmd = quad.params["minCmd"]

#     cmd = np.zeros([4, 1])
#     cmd[0] = throttle + pCmd + qCmd - rCmd
#     cmd[1] = throttle - pCmd + qCmd + rCmd
#     cmd[2] = throttle - pCmd - qCmd - rCmd
#     cmd[3] = throttle + pCmd - qCmd + rCmd
    
#     cmd[0] = min(max(cmd[0], minCmd), maxCmd)
#     cmd[1] = min(max(cmd[1], minCmd), maxCmd)
#     cmd[2] = min(max(cmd[2], minCmd), maxCmd)
#     cmd[3] = min(max(cmd[3], minCmd), maxCmd)
    
#     # Add Exponential to command
#     # ---------------------------
#     cmd = expoCmd(quad.params, cmd)

#     return cmd

# def expoCmd(params, cmd):
#     if params["ifexpo"]:
#         cmd = np.sqrt(cmd)*10
    
#     return cmd

# def expoCmdInv(params, cmd):
#     if params["ifexpo"]:
#         cmd = (cmd/10)**2
    
#     return cmd
