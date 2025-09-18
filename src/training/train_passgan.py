import os
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

# Import local modules
from src.models.passgan import Generator, Discriminator
from src.utils.data_loader import load_dataset, PasswordDataset
from src.utils.ngram_model import NgramLanguageModel
from src.utils.password_utils import gradient_penalty, generate_samples


def train_passgan(config_path):
    """Main training function for the PassGAN model"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = config['paths']

    # Parameter settings
    training_data = paths['training_data']  # User should set this
    output_dir = paths['output_dir']
    batch_size = 64
    seq_length = 16
    layer_dim = 128
    critic_iters = 10
    lamb = 10
    iters = 25000
    save_every = 1000

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)

    # Load dataset
    lines, charmap, inv_charmap = load_dataset(training_data, seq_length)

    # Save character mapping
    with open(os.path.join(paths['processed_dir'], 'charmap.pkl'), 'wb') as f:
        pickle.dump(charmap, f)
    with open(os.path.join(paths['processed_dir'], 'inv_charmap.pkl'), 'wb') as f:
        pickle.dump(inv_charmap, f)

    # Create dataset
    dataset = PasswordDataset(lines, charmap)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    n_chars = len(charmap)
    generator = Generator(seq_length, layer_dim, n_chars).to(device)
    discriminator = Discriminator(seq_length, layer_dim, n_chars).to(device)

    # Optimizers
    g_optim = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optim = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # Training loop
    data_iter = iter(dataloader)
    history = {'d_loss': [], 'g_loss': []}

    for iteration in tqdm(range(iters), total=iters):
        # Get real data
        try:
            real_data = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            real_data = next(data_iter)

        real_data = real_data.to(device).permute(0, 2, 1)  # [batch, features, seq]

        # Train discriminator
        for _ in range(critic_iters):
            # Generate fake data
            noise = torch.randn(batch_size, 128, device=device)
            with torch.no_grad():
                fake_data = generator(noise).permute(0, 2, 1)

            # Discriminator forward pass
            d_real = discriminator(real_data)
            d_fake = discriminator(fake_data)

            # Calculate loss
            d_cost = d_fake.mean() - d_real.mean()
            gp = gradient_penalty(discriminator, real_data, fake_data, batch_size, lamb, device)
            d_loss = d_cost + gp

            # Update discriminator
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

        # Train generator
        noise = torch.randn(batch_size, 128, device=device)
        fake_data = generator(noise).permute(0, 2, 1)
        d_fake = discriminator(fake_data)
        g_loss = -d_fake.mean()

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Record history
        history['d_loss'].append(d_loss.item())
        history['g_loss'].append(g_loss.item())

        # Save samples and models periodically
        if iteration % 100 == 0:
            # Generate samples
            samples = generate_samples(generator, 100, seq_length, n_chars, inv_charmap, device)
            with open(os.path.join(output_dir, 'samples', f'samples_{iteration}.txt'), 'w') as f:
                f.write('\n'.join(samples))

            # Plot loss curves
            plt.figure(figsize=(12, 6))
            plt.plot(history['d_loss'], label='Discriminator Loss')
            plt.plot(history['g_loss'], label='Generator Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training Loss')
            plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
            plt.close()

        if iteration % save_every == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'iteration': iteration,
            }, os.path.join(output_dir, 'checkpoints', f'model_{iteration}.pth'))

    # Final save
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, os.path.join(paths['model_dir'], 'final_model.pth'))

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PassGAN model')
    parser.add_argument('--config', type=str, default='config/paths.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    train_passgan(args.config)