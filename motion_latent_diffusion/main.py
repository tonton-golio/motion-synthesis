
import argparse
from utils import load_config


if __name__ == "__main__":
    # add arguments for model and mode
    parser = argparse.ArgumentParser(description='Run the model')
    parser.add_argument('--model', type=str, default='VAE', help='Model to run')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run')
    args = parser.parse_args()
    print(args.model, args.mode)
    assert args.model in ['VAE', 'imageDiffusion', 'latentDiffusion']  # assert valid
    assert args.mode in ['train', 'build', 'inference', 'optuna']

    
    if args.model == 'VAE':
        from scripts.VAE_train import train
        train(model_name='VAE4')

    elif args.model == 'imageDiffusion':
        pass