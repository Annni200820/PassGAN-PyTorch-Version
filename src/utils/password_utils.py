import torch
import numpy as np


def gradient_penalty(discriminator, real_data, fake_data, batch_size, lamb, device):
    """Calculate gradient penalty for WGAN-GP"""
    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return gradient_penalty


def generate_samples(generator, batch_size, seq_len, n_chars, inv_charmap, device):
    """Generate password samples from the generator"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(batch_size, 128, device=device)
        samples = generator(noise)
        samples = samples.argmax(dim=2).cpu().numpy()

        decoded = []
        for sample in samples:
            decoded.append(''.join(inv_charmap[idx] for idx in sample).replace('`', ''))
    return decoded


def deduplicate_passwords(input_file, output_file):
    """Remove duplicate passwords from a file"""
    unique_passwords = set()

    try:
        # Read input file
        with open(input_file, 'r', encoding='latin-1') as f:
            for line in f:
                password = line.strip()
                if password:  # Skip empty lines
                    unique_passwords.add(password)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return
    except UnicodeDecodeError:
        print(f"Error: Encoding issue with file {input_file}, try 'latin-1' or 'utf-8' encoding")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for password in unique_passwords:
                f.write(password + '\n')
    except IOError as e:
        print(f"Error writing to output file: {e}")
        return

    # Statistics
    print(f"Processing complete!")
    print(f"Unique passwords: {len(unique_passwords)}")
    print(f"Results saved to: {output_file}")