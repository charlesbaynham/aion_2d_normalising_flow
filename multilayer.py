import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# set random seed
torch.manual_seed(42)
np.random.seed(42)

# load input and output data
df_param = pd.read_csv('data/compiled/input_data.csv')
df_vectors = pd.read_csv('data/compiled/compiled_vectors.csv')
df_gen = pd.read_csv('data/model/generated_samples.csv')

# merge the input and output data on the simulation column
merged_data = df_param.merge(df_vectors, on='simulation')

# remove rows with NaNs
merged_data = merged_data.dropna()

# extract the relevant columns
data = merged_data[['cooling_beam_detuning', 'Vz']]

# convert data to torch tensors
x_data = torch.tensor(data['Vz'].values, dtype=torch.float32).unsqueeze(1)
c_data = torch.tensor(data['cooling_beam_detuning'].values, dtype=torch.float32).unsqueeze(1)

# define a simple affine coupling layer
class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AffineCouplingLayer, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, c, print_debug=False):
        if print_debug:
            print("Forward pass input to scale_net:", c[:5])
        s = self.scale_net(c)
        if print_debug:
            print("Scale_net output before clamping:", s[:5])
        s = torch.clamp(s, min=-5, max=5)  # Clamp the output of scale_net
        if print_debug:
            print("Scale_net output after clamping:", s[:5])
        t = self.translation_net(c)
        assert not torch.isnan(s).any(), "NaNs in scale_net output"
        assert not torch.isnan(t).any(), "NaNs in translation_net output"
        z = x * torch.exp(s) + t
        log_det_jacobian = torch.sum(s, dim=1)
        assert not torch.isnan(z).any(), "NaNs in z during forward pass"
        assert not torch.isnan(log_det_jacobian).any(), "NaNs in log_det_jacobian during forward pass"
        return z, log_det_jacobian

    def inverse(self, z, c):
        s = self.scale_net(c)
        s = torch.clamp(s, min=-5, max=5)  # clamp the output of scale_net
        t = self.translation_net(c)
        x = (z - t) * torch.exp(-s)
        assert not torch.isnan(x).any(), "NaNs in x during inverse pass"
        return x

# define the conditional normalising flow model
class ConditionalNormalisingFlow(nn.Module):
    def __init__(self, base_distribution, transforms):
        super(ConditionalNormalisingFlow, self).__init__()
        self.base_distribution = base_distribution
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, c, print_debug=False):
        log_det_jacobian = 0
        for i, transform in enumerate(self.transforms):
            x, ldj = transform(x, c, print_debug=print_debug and (i == 0))
            assert not torch.isnan(x).any(), "NaNs in x during forward pass"
            assert not torch.isnan(ldj).any(), "NaNs in ldj during forward pass"
            log_det_jacobian += ldj
        return x, log_det_jacobian

    def inverse(self, z, c):
        for transform in reversed(self.transforms):
            z = transform.inverse(z, c)
            assert not torch.isnan(z).any(), "NaNs in z during inverse pass"
        return z

# define the base distribution
input_dim = 1
base_distribution = torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim))

hidden_layers = 2
hidden_dims = 10

# create the model with multiple coupling layers
transforms = [AffineCouplingLayer(input_dim, hidden_dims) for _ in range(hidden_layers)]
model = ConditionalNormalisingFlow(base_distribution, transforms)

# define the loss function
def loss_fn(x, c, print_debug=False):
    z, log_det_jacobian = model(x, c, print_debug=print_debug)
    log_prob_z = base_distribution.log_prob(z).sum(dim=1)
    return -(log_prob_z + log_det_jacobian).mean()

# train the model
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    # print debug information only for the first few epochs
    print_debug = epoch < 5
    loss = loss_fn(x_data, c_data, print_debug=print_debug)
    if torch.isnan(loss):
        print(f"NaNs in loss at epoch {epoch + 1}")
        break
    loss.backward()

    # gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# plot the loss curve
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# visualise the learned transformations
with torch.no_grad():
    z_data, _ = model(x_data, c_data)

    # plot original data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(c_data[:, 0].numpy(), x_data[:, 0].numpy(), alpha=0.5)
    plt.title('Original Data')

    # plot transformed data
    plt.subplot(1, 2, 2)
    plt.scatter(c_data[:, 0].numpy(), z_data[:, 0].numpy(), alpha=0.5)
    plt.title('Transformed Data')

    plt.savefig('original_x_transformed.png')
    plt.show()

# generate 50,000 samples conditioned on uniformly chosen cooling beam detuning values between 0 and -300
num_samples = 50000
new_context = torch.FloatTensor(num_samples, 1).uniform_(-300, 0)

with torch.no_grad():
    z_samples = base_distribution.sample((num_samples,))
    generated_samples = model.inverse(z_samples, new_context)

# save generated samples to CSV
generated_df = pd.DataFrame({
    'cooling_beam_detuning': new_context.squeeze().numpy(),
    'Vz': generated_samples.squeeze().numpy()
})

generated_df.to_csv('generated_samples.csv', index=False)
print("Generated samples saved to 'generated_samples.csv'")

# define the parameter space for cooling beam detuning
cooling_beam_detuning_range = (-300, 0)
title = 'Cooling Beam Detuning'

# define quartiles for the cooling beam detuning
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
