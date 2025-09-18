import os
import pickle
import argparse
import torch
import yaml

# Import local modules
from src.models.passgan import Generator
from src.utils.password_utils import generate_samples


def generate_passwords(config_path, num_samples=10000):
    """Generate passwords using a trained PassGAN model"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = config['paths']

    # Load character mapping
    with open(os.path.join(paths['processed_dir'], 'charmap.pkl'), 'rb') as f:
        charmap = pickle.load(f)
    with open(os.path.join(paths['processed_dir'], 'inv_charmap.pkl'), 'rb') as f:
        inv_charmap = pickle.load(f)

    seq_length = 16
    layer_dim = 128

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_chars = len(charmap)
    model = Generator(seq_length, layer_dim, n_chars).to(device)

    model_path = os.path.join(paths['model_dir'], 'final_model.pth')
    model.load_state_dict(torch.load(model_path)['generator'])
    model.eval()

    # Generate passwords
    batch_size = 64
    passwords = []
    milestone_interval = 1000000  # Print progress every 1M passwords
    next_milestone = milestone_interval
    generated_count = 0

    for i in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - i)
        noise = torch.randn(current_batch, 128, device=device)
        with torch.no_grad():
            samples = model(noise).argmax(dim=2).cpu().numpy()

        for sample in samples:
            password = ''.join(inv_charmap[idx] for idx in sample).replace('`', '')
            passwords.append(password)

        # Update counter and check milestone
        generated_count += current_batch
        if generated_count >= next_milestone:
            print(f"Generated {next_milestone:,} passwords...")
            next_milestone += milestone_interval

    # Save results
    output_file = paths['generated_passwords']
    with open(output_file, 'w') as f:
        f.write('\n'.join(passwords))

    print(f"Generated {len(passwords):,} passwords to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate passwords using PassGAN')
    parser.add_argument('--config', type=str, default='config/paths.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of passwords to generate')
    args = parser.parse_args()

    generate_passwords(args.config, args.num_samples)