from __future__ import print_function, division

import os
import numpy as np
import argparse
import matplotlib.pylab as plt
import pickle

# set random set
np.random.RandomState(seed=42)

from intralaminar import intralaminar_simulation, intralaminar_analysis, intralaminar_plt
from interlaminar import interlaminar_simulation, interlaminar_activity_analysis, plot_activity_traces, \
                         calculate_interlaminar_power_spectrum, \
                         plot_interlaminar_power_spectrum, plot_power_spectrum_neurodsp

from helper_functions import firing_rate_analysis, get_network_configuration
from calculate_rate import calculate_rate

"""
Main Python file that contains the definitions for the simulation and
calls the necessary functions depending on the passed parameters.
"""


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters for the simulation')
    parser.add_argument('-sigmaoverride',
                        type=float,
                        dest='sigmaoverride',
                        default=None,
                        help='Override sigma of the Gaussian noise for ALL populations (if None leave them as is)')
    parser.add_argument('-analysis',
                        type=str,
                        dest='analysis',
                        default='debug',
                        help='Specify type of analysis to be used')
    parser.add_argument('-debug',
                        dest='debug',
                        action='store_true',
                        help='Specify whether to generate simulations for debugging')
    parser.add_argument('-noconns',
                        dest='noconns',
                        action='store_true',
                        help='Specify whether to remove connections (DEBUG MODE ONLY!)')
    parser.add_argument('-testduration',
                        type=float,
                        dest='testduration',
                        default=1000.,
                        help='Duration of test simulation (DEBUG MODE ONLY!)')
    parser.add_argument('-dt',
                        type=float,
                        dest='dt',
                        default=2e-4,
                        help='Timestep (dt) of simulation in seconds')
    parser.add_argument('-initialrate',
                        type=float,
                        dest='initialrate',
                        default=-1,
                        help='Initial rate of test simulation, if negative, use a random value (default) (DEBUG MODE ONLY!)')
    parser.add_argument('-nogui',
                        dest='nogui',
                        action='store_true',
                        help='No gui')
    return parser.parse_args()

if __name__ == "__main__":
    args = getArguments()

    # Create folder where results will be saved
    if not os.path.isdir(args.analysis):
        os.mkdir(args.analysis)

    if args.analysis == 'debug':
        print('-----------------------')
        print('Debugging')
        print('-----------------------')
        # Call a function that plots and saves of the firing rate for the intra- and interlaminar simulation
        print('Running debug simulation/analysis with %s'%args)
        dt = args.dt
        firing_rate_analysis(args.noconns, args.testduration, args.sigmaoverride, args.initialrate, dt)

    if args.analysis == 'intralaminar':
        print('-----------------------')
        print('Intralaminar Analysis')
        print('-----------------------')
        # Define dt and the trial length
        dt = args.dt
        tstop = 25 # s
        t = np.linspace(0, int(tstop), int(tstop/dt))
        transient = 5
        # speciy number of areas that communicate with each other
        Nareas = 1

        tau, sig, J, Iext, Ibgk = get_network_configuration('intralaminar', noconns=False)
        nruns = 10

        # Note: Because of the way the way intralaminar_simulation is defined only the results for L2/3
        # will be save and used for further analysis
        layer = 'L23'
        print('    Analysing layer %s' %layer)
        # check if simulation file already exists, if not run the simulation
        simulation_file = 'intralaminar/L23_simulation.pckl'
        if not os.path.isfile(simulation_file):
            print('    Re-calculating the simulation')
            simulation = intralaminar_simulation(args.analysis, layer, Iext, Ibgk, nruns, t, dt, tstop,
                            J, tau, sig, args.sigmaoverride, Nareas)
        else:
            print('    Loading the pre-saved simulation file: %s' %simulation_file)
            picklename = os.path.join('intralaminar', layer + '_simulation.pckl')
            with open(picklename, 'rb') as file1:
                simulation = pickle.load(file1)

        psd_analysis = intralaminar_analysis(simulation, Iext, nruns, layer, dt, transient)
        intralaminar_plt(psd_analysis)

    if args.analysis == 'intralaminar_c':
        print('-----------------------')
        print('Intralaminar Analysis C')
        print('Figure 2C: Effect of input on isolated layers')
        print('-----------------------')
        
        # Define dt and the trial length
        dt = args.dt
        tstop = 25  # 25 seconds per simulation
        transient = 5  # Discard first 5 seconds
        
        # Time array
        t = np.arange(0, tstop, dt)
        
        # Connection weights (from default parameters)
        wee = 1.5
        wei = -3.25
        wie = 3.5
        wii = -2.5
        
        # Time constants
        tau_2e = 0.006
        tau_2i = 0.015
        tau_5e = 0.030
        tau_5i = 0.075
        
        # Noise
        sig_2e = 0.3
        sig_2i = 0.3
        sig_5e = 0.45
        sig_5i = 0.45
        
        # Input range to test (0 to 20 nA)
        input_range = np.arange(0, 21, 1)
        nruns = 10  # Number of runs per input
        
        print(f'    Testing {len(input_range)} input levels with {nruns} runs each')
        print(f'    Input range: {input_range[0]} to {input_range[-1]} nA')
        print(f'    Total simulations: {len(input_range) * nruns * 2} (L2/3 + L5/6)')
        print(f'    Estimated time: ~{len(input_range) * nruns * 2 * tstop / 60:.1f} minutes')
        
        # Check if simulation file already exists
        simulation_file = os.path.join(args.analysis, 'simulation.pckl')
        
        if not os.path.isfile(simulation_file):
            print('    Re-calculating the simulation')
            from intralaminar import intralaminar_c_simulation
            simulation = intralaminar_c_simulation(
                args.analysis, t, dt, tstop, 
                wee, wei, wie, wii,
                tau_2e, tau_2i, tau_5e, tau_5i,
                sig_2e, sig_2i, sig_5e, sig_5i,
                input_range, nruns
            )
        else:
            print('    Loading the pre-saved simulation file: %s' % simulation_file)
            with open(simulation_file, 'rb') as f:
                simulation = pickle.load(f)
        
        # Analyze
        from intralaminar import intralaminar_c_analysis, intralaminar_c_plot
        results = intralaminar_c_analysis(simulation, input_range, nruns, dt, transient)
        
        # Plot
        fig = intralaminar_c_plot(results, args.analysis)
        
        print('    Done!')


    if args.analysis == 'interlaminar_a':
        print('-----------------------')
        print('Interlaminar Analysis')
        print('-----------------------')
        # Calculates the power spectrum for the coupled and uncoupled case for L2/3 and L5/6
        dt = args.dt
        tstop = 600
        transient = 10

        # specify number of areas that communicate with each other
        Nareas = 1
        # Note: np.arange excludes the stop so we add dt to include the last value
        t = np.arange(0, tstop, dt)

        tau, sig, J_conn, Iext_conn, Ibgk_conn = get_network_configuration('interlaminar_a', noconns=False)
        Nbin = 100 # pick one very 'bin' points

        # Calculate the rate
        rate_conn = interlaminar_simulation(args.analysis, t, dt, tstop, J_conn, tau, sig, Iext_conn, Ibgk_conn, args.sigmaoverride, Nareas)
        pxx_coupled_l23_bin, fxx_coupled_l23_bin, pxx_coupled_l56_bin, fxx_coupled_l56_bin = \
                            calculate_interlaminar_power_spectrum(rate_conn, dt, transient, Nbin)

        # Run simulation when the two layers are uncoupled
        tau, sig, J_noconn, Iext_noconn, Ibgk_noconn = get_network_configuration('interlaminar_u', noconns=False)

        rate_noconn = interlaminar_simulation(args.analysis, t, dt, tstop, J_noconn, tau, sig, Iext_noconn, Ibgk_conn, args.sigmaoverride, Nareas)
        pxx_uncoupled_l23_bin, fxx_uncoupled_l23_bin, pxx_uncoupled_l56_bin, fxx_uncoupled_l56_bin = \
                           calculate_interlaminar_power_spectrum(rate_noconn, dt, transient, Nbin)
        # Plot spectrogram
        plot_interlaminar_power_spectrum(fxx_uncoupled_l23_bin, fxx_coupled_l23_bin,
                                      pxx_uncoupled_l23_bin, pxx_coupled_l23_bin,
                                      fxx_uncoupled_l56_bin, fxx_coupled_l56_bin,
                                      pxx_uncoupled_l56_bin, pxx_coupled_l56_bin,
                                      args.analysis)

        # Plot spectrogram using neurodsp
        plot_power_spectrum_neurodsp(dt,rate_conn, rate_noconn, 'interlaminar')

        # Pickle the results rate over time
        # Transform the results so that they are saved in a dic (similar to NeuroML output)
        pyrate = {'L23_E_Py/conn': rate_conn[0, :, 0],
               'L23_I_Py/conn': rate_conn[1, :, 0],
               'L56_E_Py/conn': rate_conn[2, :, 0],
               'L56_I_Py/conn': rate_conn[3, :, 0],
               'L23_E_Py/unconn': rate_noconn[0, :, 0],
               'L23_I_Py/unconn': rate_noconn[1, :, 0],
               'L56_E_Py/unconn': rate_noconn[2, :, 0],
               'L56_I_Py/unconn': rate_noconn[3, :, 0],
               'ts': t
               }
        # picklename = os.path.join('debug', args.analysis, 'simulation.pckl')
        # if not os.path.exists(picklename):
        #     os.mkdir(os.path.dirname(picklename))
        # with open(picklename, 'wb') as filename:
        #     pickle.dump(pyrate, filename)


        print('    Done Analysis!')


    if args.analysis == 'interlaminar_b':
        print('-----------------------')
        print('Interlaminar Simulation')
        print('-----------------------')
        # Calculates the spectogram and 30 traces of actvity in layer 5/6
        # Define dt and the trial length
        dt = args.dt
        tstop = 6000
        transient = 10
        # specify number of areas that communicate with each other
        Nareas = 1
        # Note: np.arange excludes the stop so we add dt to include the last value
        t = np.arange(dt+transient, tstop + dt, dt)

        tau, sig, J, Iext, Ibgk = get_network_configuration('interlaminar_b', noconns=False)
        # frequencies of interest
        min_freq5 = 4 # alpha range
        min_freq2 = 30 # gama range

        # check if file with simulation exists, if not calculate the simulation
        simulation_file = os.path.join(args.analysis, 'simulation.pckl')
        if not os.path.isfile(simulation_file):
            print('    Re-calculating the simulation')
            rate = interlaminar_simulation(args.analysis, t, dt, tstop, J, tau, sig, Iext, Ibgk, args.sigmaoverride, Nareas)
        else:
            print('    Loading the pre-saved simulation file: %s' %simulation_file)
            with open(simulation_file, 'rb') as filename:
                rate = pickle.load(filename)

        # Analyse and Plot traces of activity in layer 5/6
        segment5, segment2, segindex, numberofzones = interlaminar_activity_analysis(rate, transient, dt, t, min_freq5)
        plot_activity_traces(dt, segment5, segindex, args.analysis)

        # Analyse and Plot spectrogram of layer L2/3
        # For now, ignore this function as I cannot generate the correct output
        # from interlaminar import interlaminar_analysis_periodeogram
        # ff, tt, Sxx = interlaminar_analysis_periodeogram(rate, segment2, transient, dt, min_freq2, numberofzones)
        # plot_spectrogram(ff, tt, Sxx)

    
    if args.analysis == 'interlaminar_c':
        print('-----------------------')
        print('Interlaminar Analysis C')
        print('Figure 3C: Effect of L5/6 input on both layers')
        print('-----------------------')
        
        # Define dt and the trial length
        dt = args.dt
        tstop = 50  # 25 seconds per simulation
        transient = 5  # Discard first 5 seconds
        
        # Specify number of areas
        Nareas = 1
        t = np.arange(0, tstop, dt)
        
        # Get network configuration for COUPLED interlaminar circuit
        tau, sig, J, Iext, Ibgk = get_network_configuration('interlaminar_b', noconns=False)
        
        # Input range to test (input to L5/6E)
        input_range = np.arange(3, 13, 1)  # 3 to 12 nA
        nruns = 10  # Number of runs per input
        
        print(f'    Testing {len(input_range)} input levels with {nruns} runs each')
        print(f'    Input range: {input_range[0]} to {input_range[-1]} nA')
        print(f'    Total simulations: {len(input_range) * nruns} = ~{len(input_range) * nruns * tstop / 60:.1f} minutes')
        
        # Check if simulation file already exists
        simulation_file = os.path.join(args.analysis, 'simulation.pckl')
        
        if not os.path.isfile(simulation_file):
            print('    Re-calculating the simulation')
            from interlaminar import interlaminar_c_simulation
            simulation = interlaminar_c_simulation(args.analysis, t, dt, tstop, J, tau, sig,
                                                   input_range, Ibgk, args.sigmaoverride, 
                                                   Nareas, nruns)
        else:
            print('    Loading the pre-saved simulation file: %s' % simulation_file)
            with open(simulation_file, 'rb') as f:
                simulation = pickle.load(f)
        
        # Analyze
        from interlaminar import interlaminar_c_analysis, interlaminar_c_plot
        results = interlaminar_c_analysis(simulation, input_range, nruns, dt, transient)
        
        # Plot
        fig = interlaminar_c_plot(results, args.analysis)
        
        print('    Done!')


    if args.analysis == 'interareal':
        print('-----------------------')
        print('Interareal Analysis')
        print('Figure 4: Microstimulation experiments (V1 ↔ V4)')
        print('-----------------------')
        
        from interareal import interareal_analysis, interareal_plt
        
        # Parameters
        dt = args.dt
        tstop = 1000  # 1000 seconds (long simulation for statistics)
        transient = 10  # Discard first 10 seconds
        t = np.arange(0, tstop, dt)
        
        # Two areas
        Nareas = 2
        areas = ['V1', 'V4']
        
        # Get base configuration
        tau, sig, J, _, Ibgk = get_network_configuration('interareal', noconns=False)
        
        # Frequency ranges
        minfreq_l23 = 30  # Gamma
        minfreq_l56 = 4   # Alpha
        
        # Statistical runs
        nstats = 10
        
        print(f'    Areas: {areas}')
        print(f'    Duration: {tstop}s per run')
        print(f'    Runs: {nstats}')
        print(f'    Total time: ~{2 * nstats * tstop / 60:.1f} minutes')
        
        # Create directory
        if not os.path.isdir('interareal'):
            os.makedirs('interareal')
        
        # ====================================================================
        # BUILD INTERAREAL CONNECTIVITY MATRIX W
        # ====================================================================
        # W[target_area, source_area, target_pop, source_pop]
        W = np.zeros((Nareas, Nareas, 4, 4))
        
        # Feedforward: V1 → V4 (L2/3 to L2/3, gamma channel)
        W[1, 0, 0, 0] = 1.0  # V4-L2/3E ← V1-L2/3E
        
        # Feedback: V4 → V1 (L5/6 to multiple layers, alpha channel)
        s = 0.5  # Partition parameter
        W[0, 1, 0, 2] = s       # V1-L2/3E ← V4-L5/6E
        W[0, 1, 1, 2] = 0.5     # V1-L2/3I ← V4-L5/6E
        W[0, 1, 2, 2] = (1-s)   # V1-L5/6E ← V4-L5/6E
        W[0, 1, 3, 3] = 0.5     # V1-L5/6I ← V4-L5/6I
        
        # Global coupling
        Gw = 0.5
        
        # ====================================================================
        # EXPERIMENT 1: Stimulate V1 (test feedforward)
        # ====================================================================
        print('\n    Experiment 1: Stimulate V1 → Measure V4')
        
        stim_v1_rest_file = 'interareal/stim_v1_rest.pckl'
        stim_v1_stim_file = 'interareal/stim_v1_stim.pckl'
        
        if os.path.isfile(stim_v1_rest_file) and os.path.isfile(stim_v1_stim_file):
            print('      Loading saved data...')
            with open(stim_v1_rest_file, 'rb') as f:
                rate_v1_rest = pickle.load(f)
            with open(stim_v1_stim_file, 'rb') as f:
                rate_v1_stim = pickle.load(f)
        else:
            print('      Running simulations (this will take ~30-40 minutes)...')
            
            # Storage: [areas*4, timepoints, 1, nstats]
            rate_v1_rest = np.zeros((Nareas * 4, len(t), 1, nstats))
            rate_v1_stim = np.zeros((Nareas * 4, len(t), 1, nstats))
            
            for stat in range(nstats):
                print(f'        Run {stat+1}/{nstats}... ', end='', flush=True)
                
                # REST: Baseline input to both areas
                Iext_rest = np.array([6, 0, 8, 0])
                rate_rest = calculate_rate(t, dt, tstop, J, tau, sig,
                                          Iext_rest, Ibgk, args.sigmaoverride,
                                          Nareas, W=W, Gw=Gw)
                
                # Reshape: [4, time, areas] → [areas*4, time, 1]
                for area in range(Nareas):
                    for pop in range(4):
                        rate_v1_rest[area * 4 + pop, :, 0, stat] = rate_rest[pop, :, area]
                
                # STIMULATION: Extra input to V1
                Iext_stim = np.array([10, 0, 8, 0])  # +4 to V1-L2/3E
                rate_stim = calculate_rate(t, dt, tstop, J, tau, sig,
                                          Iext_stim, Ibgk, args.sigmaoverride,
                                          Nareas, W=W, Gw=Gw)
                
                for area in range(Nareas):
                    for pop in range(4):
                        rate_v1_stim[area * 4 + pop, :, 0, stat] = rate_stim[pop, :, area]
                
                print('✓')
            
            # Save
            print('      Saving...')
            with open(stim_v1_rest_file, 'wb') as f:
                pickle.dump(rate_v1_rest, f)
            with open(stim_v1_stim_file, 'wb') as f:
                pickle.dump(rate_v1_stim, f)
        
        # Analyze
        print('      Analyzing...')
        px20_v1, px2_v1, px50_v1, px5_v1, fx2_v1, pgamma_v1, palpha_v1 = \
            interareal_analysis(rate_v1_rest, rate_v1_stim, transient, dt,
                              minfreq_l23, minfreq_l56, Nareas, nstats)
        
        print('      Plotting...')
        interareal_plt(areas, px20_v1, px2_v1, px50_v1, px5_v1, fx2_v1,
                      'stimulate_V1', nstats)
        
        print(f'      p-values: gamma={pgamma_v1:.6f}, alpha={palpha_v1:.6f}')
        
        # ====================================================================
        # EXPERIMENT 2: Stimulate V4 (test feedback)
        # ====================================================================
        print('\n    Experiment 2: Stimulate V4 → Measure V1')
        
        stim_v4_rest_file = 'interareal/stim_v4_rest.pckl'
        stim_v4_stim_file = 'interareal/stim_v4_stim.pckl'
        
        if os.path.isfile(stim_v4_rest_file) and os.path.isfile(stim_v4_stim_file):
            print('      Loading saved data...')
            with open(stim_v4_rest_file, 'rb') as f:
                rate_v4_rest = pickle.load(f)
            with open(stim_v4_stim_file, 'rb') as f:
                rate_v4_stim = pickle.load(f)
        else:
            print('      Running simulations (this will take ~30-40 minutes)...')
            
            rate_v4_rest = np.zeros((Nareas * 4, len(t), 1, nstats))
            rate_v4_stim = np.zeros((Nareas * 4, len(t), 1, nstats))
            
            for stat in range(nstats):
                print(f'        Run {stat+1}/{nstats}... ', end='', flush=True)
                
                # REST
                Iext_rest = np.array([6, 0, 8, 0])
                rate_rest = calculate_rate(t, dt, tstop, J, tau, sig,
                                          Iext_rest, Ibgk, args.sigmaoverride,
                                          Nareas, W=W, Gw=Gw)
                
                for area in range(Nareas):
                    for pop in range(4):
                        rate_v4_rest[area * 4 + pop, :, 0, stat] = rate_rest[pop, :, area]
                
                # STIMULATION: Extra input to V4
                Iext_stim = np.array([6, 0, 12, 0])  # +4 to V4-L5/6E
                rate_stim = calculate_rate(t, dt, tstop, J, tau, sig,
                                          Iext_stim, Ibgk, args.sigmaoverride,
                                          Nareas, W=W, Gw=Gw)
                
                for area in range(Nareas):
                    for pop in range(4):
                        rate_v4_stim[area * 4 + pop, :, 0, stat] = rate_stim[pop, :, area]
                
                print('✓')
            
            # Save
            print('      Saving...')
            with open(stim_v4_rest_file, 'wb') as f:
                pickle.dump(rate_v4_rest, f)
            with open(stim_v4_stim_file, 'wb') as f:
                pickle.dump(rate_v4_stim, f)
        
        # Analyze
        print('      Analyzing...')
        px20_v4, px2_v4, px50_v4, px5_v4, fx2_v4, pgamma_v4, palpha_v4 = \
            interareal_analysis(rate_v4_rest, rate_v4_stim, transient, dt,
                              minfreq_l23, minfreq_l56, Nareas, nstats)
        
        print('      Plotting...')
        interareal_plt(areas, px20_v4, px2_v4, px50_v4, px5_v4, fx2_v4,
                      'stimulate_V4', nstats)
        
        print(f'      p-values: gamma={pgamma_v4:.6f}, alpha={palpha_v4:.6f}')
        
        print('\n    Done! Check interareal/ folder for figures.')


    if not args.nogui:
        plt.show()

