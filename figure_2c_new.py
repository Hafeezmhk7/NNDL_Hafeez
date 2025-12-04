"""
Figure 2C with Paper-Style Shaded Regions + PICKLE SUPPORT
Saves simulation data so you don't have to re-run every time!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from calculate_rate import calculate_rate
import pickle
import os

print("="*70)
print("Figure 2C - Paper Style with Shaded Regions + PICKLE")
print("="*70)

# Exact parameters from repository
dt = 2e-4
tstop = 25
t = np.arange(0, tstop, dt)
transient = 5
transient_idx = int(transient / dt)

# Connection weights
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

# Input range and runs
input_range = np.arange(0, 21, 1)
nruns = 10

print(f"\nParameters:")
print(f"  Inputs: {input_range[0]} to {input_range[-1]} nA")
print(f"  Runs per input: {nruns}")
print(f"  Simulation: {tstop}s (discard first {transient}s)")

# ============================================================================
# PICKLE SUPPORT - CHECK IF SIMULATION ALREADY EXISTS
# ============================================================================

os.makedirs('intralaminar', exist_ok=True)
pickle_file_l23 = 'intralaminar/figure2c_L23_simulation.pckl'
pickle_file_l56 = 'intralaminar/figure2c_L56_simulation.pckl'

# ============================================================================
# L2/3 SIMULATION (Gamma)
# ============================================================================

print("\n" + "="*70)
print("L2/3 LAYER (Gamma oscillations)")
print("="*70)

J_23 = np.array([[wee, wei, 0, 0],
                 [wie, wii, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]])
tau_23 = np.array([tau_2e, tau_2i, 0.001, 0.001])
sig_23 = np.array([sig_2e, sig_2i, 0, 0])

# Check if L2/3 simulation already exists
if os.path.isfile(pickle_file_l23):
    print("✓ Loading pre-saved L2/3 simulation...")
    with open(pickle_file_l23, 'rb') as f:
        l23_simulation = pickle.load(f)
    print("✓ Loaded!")
else:
    print("⚙ Running L2/3 simulations (this will take ~15-20 minutes)...")
    
    l23_simulation = {}
    
    for inp in input_range:
        print(f"Input {inp:2d} nA: ", end='')
        
        l23_simulation[inp] = {
            'powers': [],
            'freqs': [],
            'rates': []
        }
        
        for run in range(nruns):
            if run % 3 == 0:
                print(f".", end='', flush=True)
            
            Iext_23 = np.array([inp, 0, 0, 0])
            Ibgk_23 = np.zeros(4)
            
            rate = calculate_rate(t, dt, tstop, J_23, tau_23, sig_23, 
                                 Iext_23, Ibgk_23, None, 1)
            
            rate_trace = rate[0, transient_idx:, 0]
            
            # Store the rate trace
            l23_simulation[inp]['rates'].append(rate_trace)
            
            # PSD
            fs = 1 / dt
            freqs_all, psd = signal.welch(rate_trace, fs=fs, 
                                         nperseg=min(4096, len(rate_trace)))
            
            # Find gamma peak (30-70 Hz)
            gamma_idx = np.where((freqs_all >= 30) & (freqs_all <= 70))[0]
            if len(gamma_idx) > 0:
                peak_idx = gamma_idx[np.argmax(psd[gamma_idx])]
                l23_simulation[inp]['powers'].append(psd[peak_idx])
                l23_simulation[inp]['freqs'].append(freqs_all[peak_idx])
            else:
                l23_simulation[inp]['powers'].append(0)
                l23_simulation[inp]['freqs'].append(0)
        
        print(f" ✓")
    
    # Save the simulation
    print("\n⚙ Saving L2/3 simulation to pickle file...")
    with open(pickle_file_l23, 'wb') as f:
        pickle.dump(l23_simulation, f)
    print(f"✓ Saved to: {pickle_file_l23}")

# Calculate statistics from loaded/simulated data
l23_mean_powers = []
l23_std_powers = []
l23_mean_freqs = []
l23_std_freqs = []

for inp in input_range:
    l23_mean_powers.append(np.mean(l23_simulation[inp]['powers']))
    l23_std_powers.append(np.std(l23_simulation[inp]['powers']))
    l23_mean_freqs.append(np.mean(l23_simulation[inp]['freqs']))
    l23_std_freqs.append(np.std(l23_simulation[inp]['freqs']))

# ============================================================================
# L5/6 SIMULATION (Alpha)
# ============================================================================

print("\n" + "="*70)
print("L5/6 LAYER (Alpha oscillations)")
print("="*70)

J_56 = np.array([[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, wee, wei],
                 [0, 0, wie, wii]])
tau_56 = np.array([0.001, 0.001, tau_5e, tau_5i])
sig_56 = np.array([0, 0, sig_5e, sig_5i])

# Check if L5/6 simulation already exists
if os.path.isfile(pickle_file_l56):
    print("✓ Loading pre-saved L5/6 simulation...")
    with open(pickle_file_l56, 'rb') as f:
        l56_simulation = pickle.load(f)
    print("✓ Loaded!")
else:
    print("⚙ Running L5/6 simulations (this will take ~15-20 minutes)...")
    
    l56_simulation = {}
    
    for inp in input_range:
        print(f"Input {inp:2d} nA: ", end='')
        
        l56_simulation[inp] = {
            'powers': [],
            'freqs': [],
            'rates': []
        }
        
        for run in range(nruns):
            if run % 3 == 0:
                print(f".", end='', flush=True)
            
            Iext_56 = np.array([0, 0, inp, 0])
            Ibgk_56 = np.zeros(4)
            
            rate = calculate_rate(t, dt, tstop, J_56, tau_56, sig_56, 
                                 Iext_56, Ibgk_56, None, 1)
            
            rate_trace = rate[2, transient_idx:, 0]
            
            # Store the rate trace
            l56_simulation[inp]['rates'].append(rate_trace)
            
            # PSD
            fs = 1 / dt
            freqs_all, psd = signal.welch(rate_trace, fs=fs, 
                                         nperseg=min(4096, len(rate_trace)))
            
            # Find alpha peak (8-15 Hz)
            alpha_idx = np.where((freqs_all >= 8) & (freqs_all <= 15))[0]
            if len(alpha_idx) > 0:
                peak_idx = alpha_idx[np.argmax(psd[alpha_idx])]
                l56_simulation[inp]['powers'].append(psd[peak_idx])
                l56_simulation[inp]['freqs'].append(freqs_all[peak_idx])
            else:
                l56_simulation[inp]['powers'].append(0)
                l56_simulation[inp]['freqs'].append(0)
        
        print(f" ✓")
    
    # Save the simulation
    print("\n⚙ Saving L5/6 simulation to pickle file...")
    with open(pickle_file_l56, 'wb') as f:
        pickle.dump(l56_simulation, f)
    print(f"✓ Saved to: {pickle_file_l56}")

# Calculate statistics from loaded/simulated data
l56_mean_powers = []
l56_std_powers = []
l56_mean_freqs = []
l56_std_freqs = []

for inp in input_range:
    l56_mean_powers.append(np.mean(l56_simulation[inp]['powers']))
    l56_std_powers.append(np.std(l56_simulation[inp]['powers']))
    l56_mean_freqs.append(np.mean(l56_simulation[inp]['freqs']))
    l56_std_freqs.append(np.std(l56_simulation[inp]['freqs']))

# Convert to numpy arrays
l23_mean_powers = np.array(l23_mean_powers)
l23_std_powers = np.array(l23_std_powers)
l23_mean_freqs = np.array(l23_mean_freqs)
l23_std_freqs = np.array(l23_std_freqs)

l56_mean_powers = np.array(l56_mean_powers)
l56_std_powers = np.array(l56_std_powers)
l56_mean_freqs = np.array(l56_mean_freqs)
l56_std_freqs = np.array(l56_std_freqs)

# ============================================================================
# PLOT FIGURE 2C - PAPER STYLE WITH SHADED REGIONS
# ============================================================================

print("\n" + "="*70)
print("CREATING FIGURE 2C - PAPER STYLE")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Define colors matching the paper
color_l23_power = '#2ecc71'
color_l23_freq = '#3498db'
color_l56_power = '#e67e22'
color_l56_freq = '#f39c12'

# TOP-LEFT: L2/3 Gamma Power
ax = axes[0, 0]
ax.fill_between(input_range, 
                l23_mean_powers - l23_std_powers,
                l23_mean_powers + l23_std_powers,
                alpha=0.3, color=color_l23_power, linewidth=0)
ax.plot(input_range, l23_mean_powers, 
        color=color_l23_power, linewidth=2.5, marker='o', 
        markersize=5, markerfacecolor='white', markeredgewidth=2)
ax.set_xlabel('Input to L2/3E', fontsize=12, fontweight='bold')
ax.set_ylabel('L2/3E γ power', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax.set_xlim([0, 20])
ax.set_ylim([0, 0.04])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# TOP-RIGHT: L2/3 Gamma Frequency
ax = axes[0, 1]
ax.fill_between(input_range, 
                l23_mean_freqs - l23_std_freqs,
                l23_mean_freqs + l23_std_freqs,
                alpha=0.3, color=color_l23_freq, linewidth=0)
ax.plot(input_range, l23_mean_freqs, 
        color=color_l23_freq, linewidth=2.5, marker='o', 
        markersize=5, markerfacecolor='white', markeredgewidth=2)
ax.set_xlabel('Input to L2/3E', fontsize=12, fontweight='bold')
ax.set_ylabel('L2/3E frequency (Hz)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax.set_xlim([0, 20])
ax.set_ylim([30, 60])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# BOTTOM-LEFT: L5/6 Alpha Power
ax = axes[1, 0]
ax.fill_between(input_range, 
                l56_mean_powers - l56_std_powers,
                l56_mean_powers + l56_std_powers,
                alpha=0.3, color=color_l56_power, linewidth=0)
ax.plot(input_range, l56_mean_powers, 
        color=color_l56_power, linewidth=2.5, marker='o', 
        markersize=5, markerfacecolor='white', markeredgewidth=2)
ax.set_xlabel('Input to L5/6E', fontsize=12, fontweight='bold')
ax.set_ylabel('L5/6E α power', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax.set_xlim([0, 20])
ax.set_ylim([0, 0.4])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# BOTTOM-RIGHT: L5/6 Alpha Frequency
ax = axes[1, 1]
ax.fill_between(input_range, 
                l56_mean_freqs - l56_std_freqs,
                l56_mean_freqs + l56_std_freqs,
                alpha=0.3, color=color_l56_freq, linewidth=0)
ax.plot(input_range, l56_mean_freqs, 
        color=color_l56_freq, linewidth=2.5, marker='o', 
        markersize=5, markerfacecolor='white', markeredgewidth=2)
ax.set_xlabel('Input to L5/6E', fontsize=12, fontweight='bold')
ax.set_ylabel('L5/6E frequency (Hz)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax.set_xlim([0, 20])
ax.set_ylim([6, 12])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('Figure 2C: Effect of Input on Power and Frequency', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

plt.savefig('intralaminar/figure2c_paper_style.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: intralaminar/figure2c_paper_style.png")

plt.show()

# Summary Statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print("\nL2/3 Gamma:")
print(f"  Power:     {l23_mean_powers.min():.6f} - {l23_mean_powers.max():.6f}")
print(f"  Frequency: {l23_mean_freqs.min():.1f} - {l23_mean_freqs.max():.1f} Hz")
print("\nL5/6 Alpha:")
print(f"  Power:     {l56_mean_powers.min():.6f} - {l56_mean_powers.max():.6f}")
print(f"  Frequency: {l56_mean_freqs.min():.1f} - {l56_mean_freqs.max():.1f} Hz")
print(f"\nAlpha/Gamma power ratio: {l56_mean_powers.max()/l23_mean_powers.max():.1f}x")
print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
# ```

# ---

# ## Now With Pickle - First vs Second Run

# ### **First Run** (no pickle exists):
# ```
# $ python figure2c_with_pickle.py

# ⚙ Running L2/3 simulations (this will take ~15-20 minutes)...
# Input  0 nA: ... ✓
# Input  1 nA: ... ✓
# ...
# ✓ Saved to: intralaminar/figure2c_L23_simulation.pckl

# ⚙ Running L5/6 simulations (this will take ~15-20 minutes)...
# ...
# ✓ Saved to: intralaminar/figure2c_L56_simulation.pckl

# Total time: ~30-40 minutes
# ```

# ### **Second Run** (pickle exists):
# ```
# $ python figure2c_with_pickle.py

# ✓ Loading pre-saved L2/3 simulation...
# ✓ Loaded!

# ✓ Loading pre-saved L5/6 simulation...
# ✓ Loaded!

# ✓ Saved: intralaminar/figure2c_paper_style.png

# Total time: ~10 seconds! 🚀