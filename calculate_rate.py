#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import math
from numba import jit, vectorize
# set random set
np.random.seed(42)

@vectorize
def transduction_function(element):
    if element == 0:
        return 1
    elif element <= -100:
        return 0
    else:
        return element / (1 - math.exp(-element))

@jit(nopython=True)
def dt_calculate_rate(J, rate, Ibgk, Iext, dt, tau, tstep2, xi):
    # calculate total input current
    tmp = np.dot(J, rate)
    # elementwise add elements in Ibgk, Iext, tmp
    total_input = np.add(Ibgk, np.add(Iext, tmp))
    # calculate input after the transfer function
    transfer_input = transduction_function(total_input)
    tau_r = np.divide(dt, tau)
    delta_rate = np.add(np.multiply(tau_r, (np.add(-rate, transfer_input))),
                        np.multiply(tstep2, xi))
    return delta_rate



def calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibgk, sigmaoverride, Nareas, W=1, Gw=1, initialrate=-1):
    """
    Calculates the region rate over time

    :param t:
    :param dt:
    :param tstop:
    :param J: Intra-area connectivity (4x4)
    :param tau: Membrane time constant
    :param sig: Noise level
    :param Iext: Additional current
    :param Ibgk: Background current of the system
    :param sigmaoverride: Override sigma of the Gaussian noise for ALL populations
    :param Nareas: Number of Areas to take into account
    :param W: Inter-area connectivity matrix (Nareas x Nareas x 4 x 4)
    :param Gw: Global coupling strength
    :param initialrate: Use this for t=0; if initialrate <0, use random value
    :return:
        rate: Rate over time for the areas of interest [4, timepoints, Nareas]
    """
    rate = np.zeros((4, int(round(tstop/dt) + 1), Nareas))
    
    # Apply additional input current only on excitatory layers
    sig_to_use = np.array([sigmaoverride, sigmaoverride, sigmaoverride, sigmaoverride]) if sigmaoverride!=None else sig
    tstep2 = ((dt * sig_to_use * sig_to_use) / tau) ** .5
    mean_xi = 0
    std_xi = 1
    xi = np.random.normal(mean_xi, std_xi, (4, int(round(tstop/dt)) + 1, Nareas))
    
    # Initial rate values
    rate[:, 0, :] = 5
    if initialrate>=0:
        rate[:, 0, :] = initialrate
    
    # Check if W is provided (interareal simulation)
    interareal_coupling = isinstance(W, np.ndarray) and W.ndim == 4
    
    for dt_idx in range(len(t)):
        for area in range(Nareas):
            # ====================================================================
            # INTRA-AREA DYNAMICS (within this area)
            # ====================================================================
            delta_rate = dt_calculate_rate(J, rate[:, dt_idx, area], Ibgk, Iext, 
                                          dt, tau, tstep2, xi[:, dt_idx, area])
            
            # ====================================================================
            # INTER-AREA COUPLING (from other areas) - NEW!
            # ====================================================================
            if interareal_coupling and Nareas > 1:
                # Add input from all other areas
                inter_area_input = np.zeros(4)
                
                for source_area in range(Nareas):
                    if source_area != area:  # Don't couple area to itself
                        # W[target_area, source_area, target_pop, source_pop]
                        # Sum over source populations
                        for target_pop in range(4):
                            for source_pop in range(4):
                                coupling_strength = W[area, source_area, target_pop, source_pop]
                                if coupling_strength != 0:
                                    inter_area_input[target_pop] += (
                                        Gw * coupling_strength * rate[source_pop, dt_idx, source_area]
                                    )
                
                # Add inter-area input to the rate update
                # Convert to rate change using same dynamics as intra-area
                tau_r = dt / tau
                delta_rate += tau_r * inter_area_input
            
            # ====================================================================
            # UPDATE RATE
            # ====================================================================
            rate[:, dt_idx + 1, area] = np.add(rate[:, dt_idx, area], delta_rate)
    
    # exclude the initial point that corresponds to the initial conditions
    rate = rate[:, 1:, :]
    
    return rate