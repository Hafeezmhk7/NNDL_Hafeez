import os
import numpy as np
import pickle
from scipy import signal
import matplotlib.pylab as plt

from calculate_rate import calculate_rate
from helper_functions import calculate_periodogram, compress_data, plt_filled_std, matlab_smooth


def intralaminar_analysis(simulation, Iexts, nruns, layer='L23', dt=2e-04, transient=5):
    """
    Calculates the main intralaminar analysis and dumps a pickle containing the periodogram of the analysis
    Inputs
        simulation: dictionary containing all the simulations to be analysed
        Iexts: a list of the input strengths applied on the excitatory populations
        nruns: number of simulations analysed for every Iext
        layer: Layer under analysis
        dt: time step of the simulation
        transient:

    """

    psd_dic = {}

    for Iext in Iexts:
        psd_dic[Iext] = {}

        for nrun in range(nruns):

            psd_dic[Iext][nrun] = {}
            restate = simulation[Iext][nrun]['L23_E/0/L23_E/r']

            # perform periodogram on restate.
            pxx2, fxx2 = calculate_periodogram(restate, transient, dt)

            # Compress the data by sampling every 5 points.
            bin_size = 5
            pxx_bin, fxx_bin = compress_data(pxx2, fxx2, bin_size)

            # smooth the data
            # Note: The matlab code transforms an even-window size into an odd number by subtracting by one.
            # So for simplicity I already define the window size as an odd number
            window_size = 79
            pxx = matlab_smooth(pxx_bin, window_size)

            psd_dic[Iext][nrun]['pxx'] = pxx
        # take the mean and std over the different runs
        psd_dic[Iext]['mean_pxx'] = np.mean([psd_dic[Iext][i]['pxx'] for i in range(nruns)], axis=0)
        psd_dic[Iext]['std_pxx'] = np.std([psd_dic[Iext][i]['pxx'] for i in range(nruns)], axis=0)

    # add fxx_bin to dictionary
    psd_dic['fxx_bin'] = fxx_bin

    print('    Done Analysis!')
    return psd_dic



def intralaminar_plt(psd_dic):
    # select only the first time points until fxx < 100
    fxx_plt_idx = np.where(psd_dic['fxx_bin'] < 100)
    fxx_plt = psd_dic['fxx_bin'][fxx_plt_idx]

    # find the correspondent mean and std pxx for this range
    Iexts = list(psd_dic.keys())
    # remove the fxx_bin key
    if 'fxx_bin' in Iexts:
        Iexts.remove('fxx_bin')
    for Iext in Iexts:
        psd_dic[Iext]['mean_pxx'] = psd_dic[Iext]['mean_pxx'][fxx_plt_idx]
        psd_dic[Iext]['std_pxx'] = psd_dic[Iext]['std_pxx'][fxx_plt_idx]

    # find the difference regarding the no_input
    psd_mean_0_2 = psd_dic[2]['mean_pxx'] - psd_dic[0]['mean_pxx']
    psd_mean_0_4 = psd_dic[4]['mean_pxx'] - psd_dic[0]['mean_pxx']
    psd_mean_0_6 = psd_dic[6]['mean_pxx'] - psd_dic[0]['mean_pxx']

    # find the std
    psd_std_0_2 = np.sqrt(psd_dic[2]['std_pxx'] ** 2 + psd_dic[0]['std_pxx'] ** 2)
    psd_std_0_4 = np.sqrt(psd_dic[4]['std_pxx'] ** 2 + psd_dic[0]['std_pxx'] ** 2)
    psd_std_0_6 = np.sqrt(psd_dic[6]['std_pxx'] ** 2 + psd_dic[0]['std_pxx'] ** 2)

    lcolours = ['#588ef3', '#f35858', '#bd58f3']
    fig, ax = plt.subplots(1)
    plt_filled_std(ax, fxx_plt, psd_mean_0_2, psd_std_0_2, lcolours[0], 'Input = 2')
    plt_filled_std(ax, fxx_plt, psd_mean_0_4, psd_std_0_4, lcolours[1], 'Input = 4')
    plt_filled_std(ax, fxx_plt, psd_mean_0_6, psd_std_0_6, lcolours[2], 'Input = 6')
    plt.xlim([10, 80])
    plt.ylim([0, 0.003])
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Power (resp. rest)')
    plt.legend()
    if not os.path.exists('intralaminar'):
        os.makedirs('intralaminar')
    plt.savefig('intralaminar/intralaminar.png')


def intralaminar_simulation(analysis, layer, Iexts, Ibgk, nruns, t, dt, tstop,
                            J, tau, sig, noise, Nareas):
    simulation = {}
    for Iext in Iexts:
        simulation[Iext] = {}
        Iext_a = np.array([Iext, 0, Iext, 0])
        # run each combination of external input multiple times an take the average PSD
        for nrun in range(nruns):

            simulation[Iext][nrun] = {}
            rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext_a, Ibgk, noise, Nareas)

            # Note: Save only the excitatory and inhibitory signal from L2/3.
            # For compatibility with NeuroML/LEMS transform the results into a row matrix
            simulation[Iext][nrun]['L23_E/0/L23_E/r'] = rate[0, :].reshape(-1)
            simulation[Iext][nrun]['L23_I/0/L23_I/r'] = rate[1, :].reshape(-1)

    picklename = os.path.join(analysis, layer + '_simulation.pckl')
    with open(picklename, 'wb') as file1:
        pickle.dump(simulation, file1)
    print('    Done Simulation!')
    return simulation

def intralaminar_c_simulation(analysis, t, dt, tstop, wee, wei, wie, wii, 
                              tau_2e, tau_2i, tau_5e, tau_5i,
                              sig_2e, sig_2i, sig_5e, sig_5i,
                              input_range, nruns=10):
    """
    Run separate L2/3 and L5/6 simulations with varying input
    For Figure 2C: Effect of input on power and frequency (isolated layers)
    
    Parameters:
    -----------
    analysis : str
        Analysis type (e.g., 'intralaminar_c')
    input_range : array
        Range of input currents to test
    nruns : int
        Number of runs per input level
    
    Returns:
    --------
    simulation : dict
        Dictionary containing L2/3 and L5/6 simulation results
    """
    import pickle
    import os
    from calculate_rate import calculate_rate
    
    print('    Running intralaminar simulations for Figure 2C...')
    
    # Build connectivity matrices for isolated layers
    J_23 = np.array([[wee, wei, 0, 0],
                     [wie, wii, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])
    
    J_56 = np.array([[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, wee, wei],
                     [0, 0, wie, wii]])
    
    # Time constants
    tau_23 = np.array([tau_2e, tau_2i, 0.001, 0.001])
    tau_56 = np.array([0.001, 0.001, tau_5e, tau_5i])
    
    # Noise
    sig_23 = np.array([sig_2e, sig_2i, 0, 0])
    sig_56 = np.array([0, 0, sig_5e, sig_5i])
    
    # Background input
    Ibgk = np.zeros(4)
    
    simulation = {
        'l23': {},
        'l56': {}
    }
    
    # ========================================================================
    # L2/3 Simulations
    # ========================================================================
    print('    L2/3 layer (gamma):')
    for inp in input_range:
        print(f'      Input {inp:2d} nA: ', end='')
        simulation['l23'][inp] = {}
        
        for nrun in range(nruns):
            if nrun % 3 == 0:
                print('.', end='', flush=True)
            
            Iext_23 = np.array([inp, 0, 0, 0])
            
            rate = calculate_rate(t, dt, tstop, J_23, tau_23, sig_23,
                                 Iext_23, Ibgk, None, 1)
            
            # Save L2/3E activity
            simulation['l23'][inp][nrun] = {
                'L23_E/r': rate[0, :, 0],
                'L23_I/r': rate[1, :, 0]
            }
        
        print(' ✓')
    
    # ========================================================================
    # L5/6 Simulations
    # ========================================================================
    print('    L5/6 layer (alpha):')
    for inp in input_range:
        print(f'      Input {inp:2d} nA: ', end='')
        simulation['l56'][inp] = {}
        
        for nrun in range(nruns):
            if nrun % 3 == 0:
                print('.', end='', flush=True)
            
            Iext_56 = np.array([0, 0, inp, 0])
            
            rate = calculate_rate(t, dt, tstop, J_56, tau_56, sig_56,
                                 Iext_56, Ibgk, None, 1)
            
            # Save L5/6E activity
            simulation['l56'][inp][nrun] = {
                'L56_E/r': rate[2, :, 0],
                'L56_I/r': rate[3, :, 0]
            }
        
        print(' ✓')
    
    # Save to pickle
    picklename = os.path.join(analysis, 'simulation.pckl')
    if not os.path.exists(analysis):
        os.makedirs(analysis)
    
    with open(picklename, 'wb') as f:
        pickle.dump(simulation, f)
    
    print('    Done Simulation!')
    print(f'    Saved to: {picklename}')
    
    return simulation


def intralaminar_c_analysis(simulation, input_range, nruns, dt=2e-4, transient=5):
    """
    Analyze Figure 2C: Effect of input on power and frequency
    
    Returns:
    --------
    results : dict
        Dictionary with mean and std for all metrics
    """
    from scipy import signal
    
    print('    Analyzing simulations...')
    
    transient_idx = int(transient / dt)
    fs = 1 / dt
    
    results = {
        'input_range': input_range,
        'l23_power_mean': [],
        'l23_power_std': [],
        'l23_freq_mean': [],
        'l23_freq_std': [],
        'l56_power_mean': [],
        'l56_power_std': [],
        'l56_freq_mean': [],
        'l56_freq_std': []
    }
    
    # Analyze L2/3
    for inp in input_range:
        powers = []
        freqs = []
        
        for nrun in range(nruns):
            rate_trace = simulation['l23'][inp][nrun]['L23_E/r'][transient_idx:]
            
            # PSD
            freqs_all, psd = signal.welch(rate_trace, fs=fs, 
                                         nperseg=min(4096, len(rate_trace)))
            
            # Find gamma peak (30-70 Hz)
            gamma_idx = np.where((freqs_all >= 30) & (freqs_all <= 70))[0]
            if len(gamma_idx) > 0:
                peak_idx = gamma_idx[np.argmax(psd[gamma_idx])]
                powers.append(psd[peak_idx])
                freqs.append(freqs_all[peak_idx])
            else:
                powers.append(0)
                freqs.append(0)
        
        results['l23_power_mean'].append(np.mean(powers))
        results['l23_power_std'].append(np.std(powers))
        results['l23_freq_mean'].append(np.mean(freqs))
        results['l23_freq_std'].append(np.std(freqs))
    
    # Analyze L5/6
    for inp in input_range:
        powers = []
        freqs = []
        
        for nrun in range(nruns):
            rate_trace = simulation['l56'][inp][nrun]['L56_E/r'][transient_idx:]
            
            # PSD
            freqs_all, psd = signal.welch(rate_trace, fs=fs,
                                         nperseg=min(4096, len(rate_trace)))
            
            # Find alpha peak (8-15 Hz)
            alpha_idx = np.where((freqs_all >= 8) & (freqs_all <= 15))[0]
            if len(alpha_idx) > 0:
                peak_idx = alpha_idx[np.argmax(psd[alpha_idx])]
                powers.append(psd[peak_idx])
                freqs.append(freqs_all[peak_idx])
            else:
                powers.append(0)
                freqs.append(0)
        
        results['l56_power_mean'].append(np.mean(powers))
        results['l56_power_std'].append(np.std(powers))
        results['l56_freq_mean'].append(np.mean(freqs))
        results['l56_freq_std'].append(np.std(freqs))
    
    # Convert to numpy arrays
    for key in results.keys():
        if key != 'input_range':
            results[key] = np.array(results[key])
    
    print('    Done Analysis!')
    
    return results


def intralaminar_c_plot(results, analysis):
    """
    Plot Figure 2C: Effect of input on power and frequency
    """
    import matplotlib.pyplot as plt
    import os
    
    print('    Creating Figure 2C plots...')
    
    input_range = results['input_range']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Colors matching the paper
    color_l23_power = '#2ecc71'   # Green
    color_l23_freq = '#3498db'    # Blue
    color_l56_power = '#e67e22'   # Orange
    color_l56_freq = '#f39c12'    # Yellow-orange
    
    # ========================================================================
    # TOP-LEFT: L2/3 Gamma Power
    # ========================================================================
    ax = axes[0, 0]
    
    ax.fill_between(input_range,
                    results['l23_power_mean'] - results['l23_power_std'],
                    results['l23_power_mean'] + results['l23_power_std'],
                    alpha=0.3, color=color_l23_power, linewidth=0)
    
    ax.plot(input_range, results['l23_power_mean'],
            color=color_l23_power, linewidth=2.5, marker='o',
            markersize=5, markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Input to L2/3E', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2/3E γ power', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_xlim([input_range[0], input_range[-1]])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('L2/3 Gamma Power', fontsize=11)
    
    # ========================================================================
    # TOP-RIGHT: L2/3 Gamma Frequency
    # ========================================================================
    ax = axes[0, 1]
    
    ax.fill_between(input_range,
                    results['l23_freq_mean'] - results['l23_freq_std'],
                    results['l23_freq_mean'] + results['l23_freq_std'],
                    alpha=0.3, color=color_l23_freq, linewidth=0)
    
    ax.plot(input_range, results['l23_freq_mean'],
            color=color_l23_freq, linewidth=2.5, marker='o',
            markersize=5, markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Input to L2/3E', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2/3E frequency (Hz)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_xlim([input_range[0], input_range[-1]])
    ax.set_ylim([30, 60])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('L2/3 Gamma Frequency', fontsize=11)
    
    # ========================================================================
    # BOTTOM-LEFT: L5/6 Alpha Power
    # ========================================================================
    ax = axes[1, 0]
    
    ax.fill_between(input_range,
                    results['l56_power_mean'] - results['l56_power_std'],
                    results['l56_power_mean'] + results['l56_power_std'],
                    alpha=0.3, color=color_l56_power, linewidth=0)
    
    ax.plot(input_range, results['l56_power_mean'],
            color=color_l56_power, linewidth=2.5, marker='o',
            markersize=5, markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Input to L5/6E', fontsize=12, fontweight='bold')
    ax.set_ylabel('L5/6E α power', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_xlim([input_range[0], input_range[-1]])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('L5/6 Alpha Power', fontsize=11)
    
    # ========================================================================
    # BOTTOM-RIGHT: L5/6 Alpha Frequency
    # ========================================================================
    ax = axes[1, 1]
    
    ax.fill_between(input_range,
                    results['l56_freq_mean'] - results['l56_freq_std'],
                    results['l56_freq_mean'] + results['l56_freq_std'],
                    alpha=0.3, color=color_l56_freq, linewidth=0)
    
    ax.plot(input_range, results['l56_freq_mean'],
            color=color_l56_freq, linewidth=2.5, marker='o',
            markersize=5, markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Input to L5/6E', fontsize=12, fontweight='bold')
    ax.set_ylabel('L5/6E frequency (Hz)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_xlim([input_range[0], input_range[-1]])
    ax.set_ylim([6, 12])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('L5/6 Alpha Frequency', fontsize=11)
    
    # ========================================================================
    # Final touches
    # ========================================================================
    plt.suptitle('Figure 2C: Effect of Input on Power and Frequency',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    if not os.path.exists(analysis):
        os.makedirs(analysis)
    
    output_file = os.path.join(analysis, 'figure2c.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'    ✓ Saved figure to: {output_file}')
    
    # Also save data
    data_file = os.path.join(analysis, 'figure2c_results.pckl')
    import pickle
    with open(data_file, 'wb') as f:
        pickle.dump(results, f)
    print(f'    ✓ Saved results to: {data_file}')
    
    # Summary statistics
    print('\n    Summary Statistics:')
    print('    ' + '='*50)
    print(f'    L2/3 Gamma Power:     {results["l23_power_mean"].min():.6f} - {results["l23_power_mean"].max():.6f}')
    print(f'    L2/3 Gamma Frequency: {results["l23_freq_mean"].min():.1f} - {results["l23_freq_mean"].max():.1f} Hz')
    print(f'    L5/6 Alpha Power:     {results["l56_power_mean"].min():.6f} - {results["l56_power_mean"].max():.6f}')
    print(f'    L5/6 Alpha Frequency: {results["l56_freq_mean"].min():.1f} - {results["l56_freq_mean"].max():.1f} Hz')
    print(f'    Alpha/Gamma ratio:    {results["l56_power_mean"].max()/results["l23_power_mean"].max():.1f}x')
    
    plt.show()
    
    return fig