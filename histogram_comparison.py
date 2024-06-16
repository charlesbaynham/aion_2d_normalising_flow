import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load input and output data
df_param = pd.read_csv('data/compiled/input_data.csv')
df_vectors = pd.read_csv('data/compiled/compiled_vectors.csv')
df_gen = pd.read_csv('data/model/generated_samples.csv')

# Calculate the velocity magnitude and angles for the original data
df_vectors['V_magnitude'] = np.sqrt(df_vectors['Vx']**2 + df_vectors['Vy']**2 + df_vectors['Vz']**2)
df_vectors['Theta'] = np.arccos(df_vectors['Vz'] / df_vectors['V_magnitude'])
df_vectors['Phi'] = np.arctan2(df_vectors['Vy'], df_vectors['Vx'])

# Define the parameter space for cooling beam detuning
cooling_beam_detuning_range = (-300, 0)
title = 'Cooling Beam Detuning'

# Define quartiles for the cooling beam detuning
bins = np.linspace(cooling_beam_detuning_range[0], cooling_beam_detuning_range[1], 5)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.tight_layout(pad=5.0)

def plot_histograms(ax, df, param, title, bins, normalize=False):
    for I in range(4):
        group_mask = (df[param] >= bins[I]) & (df[param] < bins[I + 1])
        filtered_data = df[group_mask]

        ax.hist(filtered_data['Vz'], bins=100, alpha=0.4, label=f'Quartile {I + 1}', density=normalize)

    ax.set_title(title)
    ax.legend()

# Plot the original data with normalization
ax = axes[0]
plot_histograms(ax, df_param.merge(df_vectors, on='simulation'), 'cooling_beam_detuning', 'Original Data (Normalised) - Cooling Beam Detuning', bins, normalize=True)

# Plot the generated data without normalization
ax = axes[1]
plot_histograms(ax, df_gen, 'cooling_beam_detuning', 'Generated Data (Normalised) - Cooling Beam Detuning', bins, normalize=True)

plt.savefig('cooling_beam_detuning_comparison.png')
plt.show()