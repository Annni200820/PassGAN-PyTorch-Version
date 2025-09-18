# PassGAN - Password Generation using GANs

This project implements a Generative Adversarial Network (GAN) for password generation, based on the original PassGAN architecture but a PyTorch Implementation.

## Project Structure

passgan-password-generation/
├── data/
│   ├── input/                 # Input data files (user provided)
│   ├── output/                # Generated passwords and results
│   └── processed/             # Processed data files
├── models/                    # Trained models
├── src/
│   ├── models/                # Model definitions
│   ├── training/              # Training scripts
│   ├── generation/            # Password generation
│   ├── analysis/              # Analysis utilities
│   └── utils/                 # Utility functions
├── config/                    # Configuration files
├── requirements.txt
├── README.md
└── .gitignore


## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Configure paths in `config/paths.yaml`:
   - Set `training_data` to your training dataset path
   - Set `test_data` to your test dataset path

## Usage

1. Train the model: 
- `python src/training/train_passgan.py --config config/paths.yaml`
2. Generate passwords:
- `python src/generation/generate_passwords.py --config config/paths.yaml --num_samples 100000`(or your number of passwords)
3. Analyze results:
- `python src/analysis/password_analysis.py --config config/paths.yaml`

## Input Data Format

The training data should be a text file with one password per line, similar to the rockyou.txt format.

## References

This implementation is based on the original PassGAN paper:
- "PassGAN: A Deep Learning Approach for Password Guessing" by Briland Hitaj et al.


