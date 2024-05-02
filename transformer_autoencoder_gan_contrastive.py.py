# -*- coding: utf-8 -*-
"""i200799_A_A03.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TmUri8rzRDHXJxeFQapP8fFvSinWgsjd
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(url)

# Print dataset information
#print("Dataset Head:")
#print(df.head())
#print("\nDataset Description:")
#print(df.describe())
#print("\nDataset Information:")
#print(df.info())

# Check for missing data
print("\nNull Values:")
print(df.isnull().sum())

# Check for duplicates and drop them
print("\nDuplicate Values:")
print(df.duplicated().sum())
df = df.drop_duplicates()

# Drop rows with missing values
df = df.dropna()

# Normalize the data
data_normalized = (df - df.mean()) / df.std()

# Print the shape of the normalized data
print("\nShape of Normalized Data:")
print(data_normalized.shape)

# Define the Transformer Autoencoder model
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerAutoencoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        return self.linear(decoded)

# Model parameters
input_dim = 9
hidden_dim = 64
num_layers = 2
num_heads = 1
num_epochs = 5
batch_size = 8

# Initialize the Transformer Autoencoder model
model = TransformerAutoencoder(input_dim, hidden_dim, num_layers, num_heads)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare data for training
data_normalized_tensor = torch.tensor(data_normalized.values, dtype=torch.float32)
data_loader = DataLoader(data_normalized_tensor, batch_size=batch_size, shuffle=True)

# Lists to store losses
epoch_losses = []

# Training loop
for epoch in range(num_epochs):
    batch_losses = []
    for batch in data_loader:
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

# Plot the training losses
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, marker='o', linestyle='-')
plt.title('Training Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Define the Generator
class Generator(nn.Module):
    def __init__(self, autoencoder_model):
        super(Generator, self).__init__()
        self.autoencoder = autoencoder_model

    def forward(self, x):
        return self.autoencoder(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Instantiate the Generator
generator = Generator(model)

# Instantiate the Discriminator
hidden_size = 128
discriminator = Discriminator(input_dim=input_dim, hidden_size=hidden_size)

# Loss function and optimizer for the discriminator
criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss function and optimizer for the generator
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Function to generate geometric mask
def generate_geometric_mask(batch_size, length, p):
    mask = torch.rand(batch_size, length) < p
    return mask.type(torch.bool)

# Function to apply mask to data
def apply_mask(data, mask):
    masked_data = data.clone()
    masked_data[mask] = 0
    return masked_data

# Data loader with masked data for contrastive learning
class AugmentedDataLoader:
    def __init__(self, data, batch_size, p):
        self.data = data
        self.batch_size = batch_size
        self.p = p

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i+self.batch_size]
            mask = generate_geometric_mask(batch.shape[1], self.p)
            masked_batch = apply_mask(batch, mask)
            yield masked_batch

# Compute summary statistics
summary_statistics = data_normalized.describe()
print("\nSummary Statistics:")
print(summary_statistics)

# Count unique classes in the target column
num_classes = len(df.iloc[:, -1].unique())  # Count unique values in the last column (target)
print("\nNumber of Classes:", num_classes)

# Contrastive Loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Initialize models and optimizers
model = TransformerAutoencoder(input_dim=9, hidden_dim=64, num_layers=2, num_heads=1)
generator = Generator(model)
discriminator = Discriminator(input_dim=9, hidden_size=hidden_size)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
ae_optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss functions
criterion_gan = nn.BCELoss()
contrastive_loss = ContrastiveLoss(margin=1.0)

# Training loop
for epoch in range(num_epochs):
    for data in data_loader:
        batch_size, sequence_length = data.shape
        mask = generate_geometric_mask(batch_size, sequence_length, p=0.1)
        positive_data = apply_mask(data, mask)
        negative_data = torch.roll(data, shifts=1, dims=0)

        # Train Discriminator
        real_output = discriminator(data)
        real_loss = criterion_gan(real_output, torch.ones_like(real_output))
        fake_data = generator(data)
        fake_output = discriminator(fake_data.detach())
        fake_loss = criterion_gan(fake_output, torch.zeros_like(fake_output))
        d_loss = real_loss + fake_loss
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        trick_output = discriminator(fake_data)
        g_loss = criterion_gan(trick_output, torch.ones_like(trick_output))
        g_loss.backward()
        g_optimizer.step()

        # Train Autoencoder with Contrastive Learning
        ae_optimizer.zero_grad()
        original_recon = model(data)
        positive_recon = model(positive_data)
        negative_recon = model(negative_data)
        pos_loss = contrastive_loss(original_recon, positive_recon, torch.ones(data.size(0), device=data.device))
        neg_loss = contrastive_loss(original_recon, negative_recon, torch.zeros(data.size(0), device=data.device))
        ae_loss = pos_loss + neg_loss
        ae_loss.backward()
        ae_optimizer.step()

        print(f'Epoch {epoch+1}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, AE Loss: {ae_loss.item():.4f}')

# Function to calculate evaluation metrics
def calculate_metrics(predicted, actual, threshold=0.1):
    mse = torch.mean((predicted - actual) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predicted - actual))
    mse = mse.item()
    rmse = rmse.item()
    mae = mae.item()
    correct_predictions = torch.abs(predicted - actual) <= threshold
    accuracy = torch.mean(correct_predictions.float()).item()
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "Accuracy": accuracy}

# Define a threshold for anomaly detection (e.g., 0.5)
threshold = 1.2

# Evaluation loop with anomaly detection
anomalies = []
mse_values = []  # List to store MSE values for visualization
for epoch in range(num_epochs):
    for data in data_loader:
        with torch.no_grad():
            predicted = model(data)
            actual = data

            # Calculate MSE for the current batch
            mse = calculate_metrics(predicted, actual)["MSE"]

            # Append MSE value to the list for visualization
            mse_values.append(mse)

            # Identify anomalies based on MSE exceeding the threshold
            if mse > threshold:
                anomalies.append((epoch, mse))  # Store epoch number and MSE for anomalies

    print(f"Epoch {epoch+1} Metrics: MSE={mse:.4f}")

# Print detected anomalies
print("Detected Anomalies:", anomalies)

# Plot MSE values with the threshold
plt.figure(figsize=(10, 5))
plt.plot(mse_values, label='MSE')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Mean Squared Error (MSE) with Anomaly Threshold')
plt.xlabel('Data Point')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()