import numpy as np
import os
import pickle
from scipy import signal, fftpack
import math
import matplotlib.pylab as plt
from neurodsp import spectral

from calculate_rate import calculate_rate
from helper_functions import calculate_periodogram, compress_data, find_peak_frequency


def interlaminar_simulation(analysis, t, dt, tstop, J, tau, sig, Iext, Ibgk, noise, Nareas):
    rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibgk, noise, Nareas)
    picklename = os.path.join(analysis, 'simulation.pckl')
    with open(picklename, 'wb') as filename:
        pickle.dump(rate, filename)
    print('    Done Simulation!')
    return rate




def my_pretransformations(x, window, noverlap, fs):

    # Place x into columns and return the corresponding central time estimates
    # restructure the data
    ncol = int(np.floor((x.shape[0] - noverlap) / (window.shape[0] - noverlap)))
    coloffsets = np.expand_dims(range(ncol), axis=0) * (window.shape[0] - noverlap)
    rowindices = np.expand_dims(range(0, window.shape[0]), axis=1)

    # segment x into individual columns with the proper offsets
    xin = x[rowindices + coloffsets]
    # return time vectors
    t = coloffsets + (window.shape[0]/2)/ fs
    return xin, t


def interlaminar_activity_analysis(rate, transient, dt, t, min_freq5):

    # Note: This analysis selects only the excitatory populations from L2/3 and L5/6
    x_2 = rate[0, int(round((transient + dt)/dt)) - 1:, 0]
    x_5 = rate[2, int(round((transient + dt)/dt)) - 1:, 0]

    pxx, fxx = calculate_periodogram(x_5, transient, dt)
    # Extract L5/6E activity
    restate = rate[2, :, 0]
    f_peakalpha, _, _, _ = find_peak_frequency(fxx, pxx, min_freq5, restate)
    print('    Average peak frequency on the alpha range: %.02f Hz' %f_peakalpha)

    # band-pass filter L5 activity
    fmin = 7; fmax = 12; fs = 1/dt
    filter_order = 3
    bf, af = signal.butter(filter_order, [fmin/(fs/2), fmax/(fs/2)], 'bandpass')
    # Note: padlen is differently defined in the scipy implementation
    # simulated LFP
    re5bp = -signal.filtfilt(bf, af, x_5, padlen=3*(max(len(af), len(bf)) - 1))

    # Locate N well spaced peaks along the trial
    tzone = 4
    # length of the tzone in indices
    tzoneindex = int(round((tzone/dt)))
    rest = len(t) % tzoneindex
    time5 = t[0:-rest]; re5 = re5bp[0:-rest]; re2 = x_2[0:-rest]
    numberofzones = int(round(len(re5)/tzoneindex))
    zones5 = np.reshape(re5, (tzoneindex, numberofzones), order='F')
    zones2 = np.reshape(re2, (tzoneindex, numberofzones), order='F')

    # find a prominent peak around the center of each zone
    # Note: slicing in matlab includes the last index
    tzi_bottom = int(round(tzoneindex/2-tzoneindex/4)) + 1
    tzi_top = int(round(tzoneindex/2+tzoneindex/4)) + 1

    alpha_peaks = np.zeros((numberofzones))
    aploc = np.zeros((numberofzones))
    # find max value for each zone
    for i in range(numberofzones):
        alpha_peaks[i] = np.max(zones5[tzi_bottom:tzi_top, i])
        # Todo: indices are shifted by two respect to matlab code
        aploc[i] = np.argmax(zones5[tzi_bottom:tzi_top, i]) + tzi_bottom

    # chose a segment of 7 cycles centered on the prominent peak of each zone
    seglength = 7/f_peakalpha
    # Check if there is any problems with the segment window
    if seglength/2 >= tzi_bottom * dt:
        print('Problems with segment window!')

    # segment semi-length in indices
    segindex = int(round(0.5 * seglength/dt))
    # Note: The + 2 corrects for indexing in python
    # Note: + 1 correct for inclusive range in matlab
    # Calculate the size of the resulting matrix
    segind01 = int(round(aploc[0] - segindex) + 2)
    segind02 = int(round(aploc[0] + segindex) + 2)
    segment2 = np.zeros((segind02 - segind01 + 1, numberofzones))
    segment5 = np.zeros((segind02 - segind01 + 1, numberofzones))
    for i in range(numberofzones):
        segind1 = int(round(aploc[i] - segindex) + 2)
        segind2 = int(round(aploc[i] + segindex) + 2)
        if alpha_peaks[i] >= 0.:
            segment5[:, i] = zones5[segind1:segind2 + 1, i]
            segment2[:, i] = zones2[segind1:segind2 + 1, i]
    return segment5, segment2, segindex, numberofzones


def interlaminar_analysis_periodeogram(rate, segment2, transient, dt, min_freq2, numberofzones):
    # TODO: still in construction
    # calculate the spectogram for L2/3 and average the results over the segments
    pxx2, fxx2 = calculate_periodogram(rate[0, :, 0], transient, dt)
    # Extract L5/6E activity
    restate = rate[2, :, 0]
    f_peakgamma, _, _, _ = find_peak_frequency(fxx2, pxx2, min_freq2, restate)
    print('    Average peak frequency on the gamma range: %.02f Hz' %f_peakgamma)
    timewindow = 7/f_peakgamma
    window_len = int(round(timewindow/dt))
    window = signal.get_window('hamming', window_len)
    noverlap = int(round(0.95 * window_len))
    lowest_frequency = 25; highest_frequency = 45; step_frequency = .25
    freq_displayed = np.arange(lowest_frequency, highest_frequency + step_frequency, step_frequency)
    fs = int(1/dt)

    dic = {'segment2': segment2,
           'window_len': window_len,
           'noverlap': noverlap,
           'freq_displayed': freq_displayed,
           'fs': fs}
    with open('periodogram.pckl', 'wb') as filename:
        pickle.dump(dic, filename)

    # try loading mat file with the correct input to the spectogram
    from scipy.io import loadmat
    # matfile = '../Matlab/fig3/segment2.mat'
    # mat = loadmat(matfile)
    # segment2 = mat['segment2']

    Sxx = np.zeros((freq_displayed.shape[0], 83, numberofzones), dtype=complex)
    for n in range(numberofzones):
        print(n)
        xin, t = my_pretransformations(segment2[:, n], window, noverlap, fs)
        data = np.multiply(np.expand_dims(window, axis=1), xin)

        for jj in range(data.shape[1]):
            for ii in range(freq_displayed.shape[0]):
                Sxx[ii, jj, n] = goertzel_second(data[:, jj], freq_displayed[ii], data.shape[0])

    Sxx_mean = np.mean(Sxx, axis=2)
    # perform spectrogram on results from goertzel
    # compensate for the power of the window
    U = np.dot(np.expand_dims(window, axis=0), window)
    Sxx_conj = (Sxx_mean * np.conj(Sxx_mean))/U
    # change type back to float
    Sxx_fin = Sxx_conj.astype(float)


    goertzel_second(data[:, 0], freq_displayed[0], data.shape[0])
    goerzel = compute_goertzel(freq_displayed[0], fs, data[:, 0])
    # TODO: Try to use fft instead of the goertzel algorithm to calculate the fft
    Xx = np.zeros((freq_displayed.shape[0], xin.shape[1], numberofzones), dtype=complex)
    for n in range(numberofzones):
        for i in range(xin.shape[1]):
            Xx[:, i, n] = fftpack.fft(data[:, i], freq_displayed.shape[0])


    # nfft = np.arange(lowest_frequency, highest_frequency + step_frequency, step_frequency)
    # # obtain spectograms
    # for i in range(numberofzones):
    #     # first one is the one working the best
        ff, tt, Sxx = signal.spectrogram(segment2[:, i], fs=fs, window=window, noverlap=noverlap, return_onesided=False,
                                         detrend=False, scaling='density', mode='psd')
    #     # try to get only the frequencies between 25 and 45
    #     ff, tt, Sxx = signal.spectrogram(segment2[:, i], fs=fs, nfft=nfft, return_onesided=False, detrend=False, scaling='density',
    #                                      mode='psd')
    #     ff, tt, Sxx = signal.spectrogram(segment2[:, i], fs=fs, return_onesided=True, detrend=False,
    #                                      scaling='density', mode='psd')


    print('Done Analysis!')
    return ff, tt, Sxx

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def plot_activity_traces(dt, segment5, segindex, analysis):
    # calculate the peak-centered alpha wave by averaging
    alphawaves = np.mean(segment5, axis=1)
    alphatime = [(i*dt) - (segindex*dt) for i in range(1, alphawaves.shape[0] + 1)]
    # plot the first 100 elements from segment5
    grey_rgb = (.7, .7, .7)
    plt.figure()
    plt.plot(alphatime, segment5[:, 0:100], color=grey_rgb)
    plt.plot(alphatime, alphawaves, 'b')
    plt.xlabel('Time relative to alpha peak (s)')
    plt.ylabel('LFP, L5/6')
    plt.xlim([-.24, .24])
    plt.savefig(os.path.join(analysis, 'activity_traces.png'))


def compute_goertzel(target_frequency, sampling_rate, data):

    # Number of sample points
    nsamples = data.shape[0]
    scaling_factor = nsamples / 2.0
    k = (.5 + ((nsamples * target_frequency) / sampling_rate))
    omega = (2 * math.pi * k) / nsamples
    sine = math.sin(omega)
    cosine = math.cos(omega)
    coeff = 2 * cosine
    q1 = 0; q2 = 0

    for i in range(nsamples):
        q0 = coeff * q1 - q2 + data[i]
        q2 = q1
        q1 = q0

    real = (q1 - q2 * cosine) / scaling_factor
    imag = (q2 * sine) / scaling_factor
    magnitude = np.sqrt(real * real + imag * imag)
    return magnitude

def goertzel_second(x, k, N):
    #k = k1/ 5000 * 821
    w = 2 * math.pi * k/ N
    cw = math.cos(w); c = 2 * cw;
    sw = math.sin(w)
    z1 = 0; z2= 0;
    for n in range(N):
        z0 = x[n] + c * z1 - z2
        z2 = z1
        z1 = z0
    real = cw * z1 - z2
    imag = sw * z1
    return complex(real, imag)

def goertzel_third(x, k1, N, f, Fs):
    k = k1/ 5000 * 821
    w = (2 * math.pi * f)/ (Fs * N**2)
    cw = math.cos(w); c = 2 * cw;
    sw = math.sin(w)
    z1 = 0; z2= 0;
    for n in range(N):
        z0 = x[n] + c * z1 - z2
        z2 = z1
        z1 = z0
    real = cw * z1 - z2
    imag = sw * z1
    return complex(real, imag)

def plot_spectrogram(ff, tt, Sxx):
    plt.figure()
    plt.pcolormesh(tt, ff, Sxx, cmap='jet')
    plt.ylim([25, 45])
    plt.show()


def calculate_interlaminar_power_spectrum(rate, dt, transient, Nbin):
    # Calculate the rate for the passed connectivity
    pxx_l23, fxx_l23 = calculate_periodogram(rate[0, :, 0], transient, dt)
    pxx_l56, fxx_l56 = calculate_periodogram(rate[2, :, 0], transient, dt)
    # Compress data by selecting one data point every "bin"
    pxx_l23_bin, fxx_l23_bin = compress_data(pxx_l23, fxx_l23, Nbin)
    pxx_l56_bin, fxx_l56_bin = compress_data(pxx_l56, fxx_l56, Nbin)
    return pxx_l23_bin, fxx_l23_bin, pxx_l56_bin, fxx_l56_bin

def plot_power_spectrum_neurodsp(dt, rate_conn, rate_noconn, analysis):
    fs = 1/dt

    # Plot the results for L23
    freq_mean_L23_conn, P_mean_L23_conn = spectral.compute_spectrum(rate_conn[0, :, 0], fs, avg_type='mean')
    freq_mean_L23_noconn, P_mean_L23_noconn = spectral.compute_spectrum(rate_noconn[0, :, 0], fs, avg_type='mean')

    plt.figure()
    plt.loglog(freq_mean_L23_conn, P_mean_L23_conn, label='Coupled', linewidth=2, color='g')
    plt.loglog(freq_mean_L23_noconn, P_mean_L23_noconn, label='Uncoupled', linewidth=2, color='k')
    plt.xlim([1, 100])
    plt.ylim([10 ** -4, 10 ** -2])
    plt.ylabel('Power')
    plt.xlabel('Frequency (Hz)')
    plt.legend()

    # Plot the results for L56
    freq_mean_L56_conn, P_mean_L56_conn = spectral.compute_spectrum(rate_conn[2, :, 0], fs, avg_type='mean')
    freq_mean_L56_noconn, P_mean_L56_noconn = spectral.compute_spectrum(rate_noconn[2, :, 0], fs, avg_type='mean')

    plt.figure()
    plt.loglog(freq_mean_L56_conn, P_mean_L56_conn, label='Coupled', linewidth=2, color='#FF7F50')
    plt.loglog(freq_mean_L56_noconn, P_mean_L56_noconn, label='Uncoupled', linewidth=2, color='k')
    plt.xlim([1, 100])
    plt.ylim([10**-5, 10**-0])
    plt.ylabel('Power')
    plt.xlabel('Frequency (Hz)')
    plt.legend()

def plot_interlaminar_power_spectrum(fxx_uncoupled_l23_bin, fxx_coupled_l23_bin,
                                  pxx_uncoupled_l23_bin, pxx_coupled_l23_bin,
                                  fxx_uncoupled_l56_bin, fxx_coupled_l56_bin,
                                  pxx_uncoupled_l56_bin, pxx_coupled_l56_bin,
                                  analysis):
    plt.figure()
    plt.loglog(fxx_uncoupled_l23_bin, pxx_uncoupled_l23_bin, 'k', label='no coupling')
    plt.loglog(fxx_coupled_l23_bin, pxx_coupled_l23_bin, 'g', label='with coupling')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('L2/3 Power')
    plt.legend()
    plt.xlim([1, 100])
    plt.ylim([10**-4, 10**-2])
    if not os.path.exists('interlaminar'):
        os.makedirs('interlaminar')
    plt.savefig(os.path.join(analysis, 'spectrogram_l23.png'))

    plt.figure()
    plt.loglog(fxx_uncoupled_l56_bin, pxx_uncoupled_l56_bin, color='k', label='no coupling')
    plt.loglog(fxx_coupled_l56_bin, pxx_coupled_l56_bin, color='#FF7F50', label='with coupling')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('L5/6 Power')
    plt.legend()
    plt.xlim([1, 100])
    plt.ylim([10**-5, 10**-0])
    plt.savefig(os.path.join(analysis, 'spectrogram_l56.png'))



def interlaminar_c_simulation(analysis, t, dt, tstop, J, tau, sig, input_range, Ibgk, sigmaoverride, Nareas, nruns=10):
    """
    Run interlaminar coupled simulation with varying input to L5/6
    For Figure 3C: Effect of L5/6 input on both layers
    
    Parameters:
    -----------
    analysis : str
        Analysis type (e.g., 'interlaminar_c')
    input_range : array
        Range of input currents to test for L5/6E
    nruns : int
        Number of runs per input level
    
    Returns:
    --------
    simulation : dict
        Dictionary containing all simulation results
    """
    import pickle
    import os
    from calculate_rate import calculate_rate
    
    print('    Running coupled L2/3-L5/6 simulations...')
    
    simulation = {}
    
    for inp in input_range:
        print(f'    Input to L5/6E = {inp:.1f} nA: ', end='')
        simulation[inp] = {}
        
        for nrun in range(nruns):
            if nrun % 3 == 0:
                print('.', end='', flush=True)
            
            simulation[inp][nrun] = {}
            
            # Input to L5/6E only (population 2)
            Iext = np.array([0, 0, inp, 0])
            
            # Run simulation
            rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibgk, sigmaoverride, Nareas)
            
            # Save both L2/3E and L5/6E activity
            simulation[inp][nrun]['L23_E/r'] = rate[0, :, 0]  # L2/3E
            simulation[inp][nrun]['L23_I/r'] = rate[1, :, 0]  # L2/3I
            simulation[inp][nrun]['L56_E/r'] = rate[2, :, 0]  # L5/6E
            simulation[inp][nrun]['L56_I/r'] = rate[3, :, 0]  # L5/6I
        
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


def interlaminar_c_analysis(simulation, input_range, nruns, dt=2e-4, transient=5):
    """
    Analyze Figure 3C: Effect of L5/6 input on both layers
    
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
        'l23_gamma_power_mean': [],
        'l23_gamma_power_std': [],
        'l23_firing_rate_mean': [],
        'l23_firing_rate_std': [],
        'l56_alpha_power_mean': [],
        'l56_alpha_power_std': [],
        'l56_firing_rate_mean': [],
        'l56_firing_rate_std': []
    }
    
    for inp in input_range:
        # Storage for this input level
        l23_gamma_powers = []
        l23_firing_rates = []
        l56_alpha_powers = []
        l56_firing_rates = []
        
        for nrun in range(nruns):
            # Get L2/3E activity
            l23_rate = simulation[inp][nrun]['L23_E/r'][transient_idx:]
            
            # Get L5/6E activity
            l56_rate = simulation[inp][nrun]['L56_E/r'][transient_idx:]
            
            # Calculate L2/3 gamma power (30-70 Hz)
            freqs, psd_l23 = signal.welch(l23_rate, fs=fs, nperseg=min(4096, len(l23_rate)))
            gamma_idx = np.where((freqs >= 30) & (freqs <= 70))[0]
            if len(gamma_idx) > 0:
                l23_gamma_powers.append(np.max(psd_l23[gamma_idx]))
            else:
                l23_gamma_powers.append(0)
            
            # Calculate L2/3 mean firing rate
            l23_firing_rates.append(np.mean(l23_rate))
            
            # Calculate L5/6 alpha power (8-15 Hz)
            freqs, psd_l56 = signal.welch(l56_rate, fs=fs, nperseg=min(4096, len(l56_rate)))
            alpha_idx = np.where((freqs >= 8) & (freqs <= 15))[0]
            if len(alpha_idx) > 0:
                l56_alpha_powers.append(np.max(psd_l56[alpha_idx]))
            else:
                l56_alpha_powers.append(0)
            
            # Calculate L5/6 mean firing rate
            l56_firing_rates.append(np.mean(l56_rate))
        
        # Store mean and std
        results['l23_gamma_power_mean'].append(np.mean(l23_gamma_powers))
        results['l23_gamma_power_std'].append(np.std(l23_gamma_powers))
        results['l23_firing_rate_mean'].append(np.mean(l23_firing_rates))
        results['l23_firing_rate_std'].append(np.std(l23_firing_rates))
        results['l56_alpha_power_mean'].append(np.mean(l56_alpha_powers))
        results['l56_alpha_power_std'].append(np.std(l56_alpha_powers))
        results['l56_firing_rate_mean'].append(np.mean(l56_firing_rates))
        results['l56_firing_rate_std'].append(np.std(l56_firing_rates))
    
    # Convert to numpy arrays
    for key in results.keys():
        if key != 'input_range':
            results[key] = np.array(results[key])
    
    print('    Done Analysis!')
    
    return results


def interlaminar_c_plot(results, analysis):
    """
    Plot Figure 3C: All 4 subplots (CORRECTED VERSION)
    """
    import matplotlib.pyplot as plt
    import os
    
    print('    Creating Figure 3C plots...')
    
    input_range = results['input_range']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Colors - CORRECTED to match paper
    color_l23 = '#2ecc71'         # Green for ALL L2/3 properties
    color_l56_power = '#e67e22'   # Orange for L5/6 alpha power
    color_relationship = '#f39c12' # Yellow-orange for bottom-right
    
    # ========================================================================
    # TOP-LEFT: L2/3E gamma power vs Input to L5/6E
    # ========================================================================
    ax = axes[0, 0]
    
    ax.fill_between(input_range,
                    results['l23_gamma_power_mean'] - results['l23_gamma_power_std'],
                    results['l23_gamma_power_mean'] + results['l23_gamma_power_std'],
                    alpha=0.3, color=color_l23, linewidth=0)
    
    ax.plot(input_range, results['l23_gamma_power_mean'],
            color=color_l23, linewidth=2.5, marker='o',
            markersize=5, markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Input to L5/6E', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2/3E γ power', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_xlim([input_range[0], input_range[-1]])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('L2/3 Gamma Power', fontsize=11)
    
    # ========================================================================
    # TOP-RIGHT: L2/3E firing rate vs Input to L5/6E
    # CORRECTED: Changed color from blue to GREEN
    # ========================================================================
    ax = axes[0, 1]
    
    ax.fill_between(input_range,
                    results['l23_firing_rate_mean'] - results['l23_firing_rate_std'],
                    results['l23_firing_rate_mean'] + results['l23_firing_rate_std'],
                    alpha=0.3, color=color_l23, linewidth=0)  # ← GREEN not blue!
    
    ax.plot(input_range, results['l23_firing_rate_mean'],
            color=color_l23, linewidth=2.5, marker='o',  # ← GREEN not blue!
            markersize=5, markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Input to L5/6E', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2/3E firing rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_xlim([input_range[0], input_range[-1]])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('L2/3 Firing Rate', fontsize=11)
    
    # ========================================================================
    # BOTTOM-LEFT: L5/6E alpha power vs Input to L5/6E
    # ========================================================================
    ax = axes[1, 0]
    
    ax.fill_between(input_range,
                    results['l56_alpha_power_mean'] - results['l56_alpha_power_std'],
                    results['l56_alpha_power_mean'] + results['l56_alpha_power_std'],
                    alpha=0.3, color=color_l56_power, linewidth=0)
    
    ax.plot(input_range, results['l56_alpha_power_mean'],
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
    # BOTTOM-RIGHT: L5/6E alpha power vs L2/3E firing rate
    # CORRECTED: SWAPPED AXES!
    # X-axis: L2/3E firing rate (was Y before)
    # Y-axis: L5/6E alpha power (was X before)
    # ========================================================================
    ax = axes[1, 1]
    
    # CORRECTED: Swap X and Y!
    ax.errorbar(results['l23_firing_rate_mean'],        # ← X-axis: L2/3 rate
                results['l56_alpha_power_mean'],         # ← Y-axis: L5/6 alpha
                xerr=results['l23_firing_rate_std'],     # ← X error bars
                yerr=results['l56_alpha_power_std'],     # ← Y error bars
                fmt='o', markersize=6, capsize=4, capthick=2,
                color=color_relationship, ecolor=color_relationship, linewidth=2)
    
    # Add trend line (CORRECTED: swapped axes)
    ax.plot(results['l23_firing_rate_mean'],
            results['l56_alpha_power_mean'],
            '--', color=color_relationship, linewidth=1.5, alpha=0.5)
    
    # CORRECTED: Swapped axis labels!
    ax.set_xlabel('L2/3E firing rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('L5/6E α power', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Inverse Relationship', fontsize=11)
    
    # Add correlation annotation (CORRECTED: use correct order)
    from scipy.stats import pearsonr
    corr, pval = pearsonr(results['l23_firing_rate_mean'],   # ← X: L2/3 rate
                          results['l56_alpha_power_mean'])    # ← Y: L5/6 alpha
    ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.3e}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========================================================================
    # Final touches
    # ========================================================================
    plt.suptitle('Figure 3C: Effect of L5/6 Input on Interlaminar Dynamics',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    if not os.path.exists(analysis):
        os.makedirs(analysis)
    
    output_file = os.path.join(analysis, 'figure3c.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'    ✓ Saved figure to: {output_file}')
    
    # Also save data
    data_file = os.path.join(analysis, 'figure3c_results.pckl')
    import pickle
    with open(data_file, 'wb') as f:
        pickle.dump(results, f)
    print(f'    ✓ Saved results to: {data_file}')
    
    plt.show()
    
    return fig