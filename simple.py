import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# set random seed
torch.manual_seed(42)
np.random.seed(42)

# load input and output data
input_data = pd.read_csv('data/compiled/input_data.csv')
output_data = pd.read_csv('data/compiled/compiled_vectors.csv')

# inspect the data to ensure no NaNs
#print("Input Data Head:\n", input_data.head())
#print("Output Data Head:\n", output_data.head())

# merge the input and output data on the simulation column
merged_data = pd.merge(output_data, input_data, on='simulation')

# remove rows with NaNs
merged_data = merged_data.dropna()

# Extract the relevant columns
data = merged_data[['cooling_beam_detuning', 'Vz']]

# convert data to torch tensors
x_data = torch.tensor(data['Vz'].values, dtype=torch.float32).unsqueeze(1)
c_data = torch.tensor(data['cooling_beam_detuning'].values, dtype=torch.float32).unsqueeze(1)

# check for NaNs in the tensors
#assert not torch.isnan(x_data).any(), "NaNs found in x_data"
#assert not torch.isnan(c_data).any(), "NaNs found in c_data"

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

    def forward(self, x, c):
        s = self.scale_net(c)
        t = self.translation_net(c)
        z = x * torch.exp(s) + t
        log_det_jacobian = torch.sum(s, dim=1)
        return z, log_det_jacobian

    def inverse(self, z, c):
        s = self.scale_net(c)
        t = self.translation_net(c)
        x = (z - t) * torch.exp(-s)
        return x

# define the conditional normalising flow model
class ConditionalNormalisingFlow(nn.Module):
    def __init__(self, base_distribution, transforms):
        super(ConditionalNormalisingFlow, self).__init__()
        self.base_distribution = base_distribution
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, c):
        log_det_jacobian = 0
        for transform in self.transforms:
            x, ldj = transform(x, c)
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

# create the model with one coupling layer
transforms = [AffineCouplingLayer(input_dim, 10)]
model = ConditionalNormalisingFlow(base_distribution, transforms)

# define the loss function
def loss_fn(x, c):
    z, log_det_jacobian = model(x, c)
    log_prob_z = base_distribution.log_prob(z).sum(dim=1)
    return -(log_prob_z + log_det_jacobian).mean()

# train the model
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_fn(x_data, c_data)
    if torch.isnan(loss):
        print(f"NaNs in loss at epoch {epoch + 1}")
        break
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

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
    plt.show()

# generate 5e4 samples conditioned on uniformly chosen cooling beam detuning values between 0 and -300
num_samples = 500000
new_context = torch.FloatTensor(num_samples, 1).uniform_(-300, 0)

with torch.no_grad():
    z_samples = base_distribution.sample((num_samples,))
    generated_samples = model.inverse(z_samples, new_context)

# save generated samples to CSV
generated_df = pd.DataFrame({
    'cooling_beam_detuning': new_context.squeeze().numpy(),
    'Vz': generated_samples.squeeze().numpy()
})

generated_df.to_csv('data/model/generated_samples.csv', index=False)
print("Generated samples saved to 'data/model/generated_samples.csv'")