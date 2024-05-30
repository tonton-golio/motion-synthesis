
import argparse
from utils import load_config


if __name__ == "__main__":
    # add arguments for model and mode
    parser = argparse.ArgumentParser(description='Run the model')
    parser.add_argument('--model', type=str, default='VAE', help='Model to run')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run')
    args = parser.parse_args()
    print(args.model, args.mode)
    assert args.model in ['VAE1', 'VAE2', 'VAE3', 'VAE4', 'VAE5',
                          'LD']  # assert valid
    assert args.mode in ['train', 'build', 'inference', 'optuna']

    
    if args.model[:3] == 'VAE':
        if args.mode == 'train':
            # train the model
            from scripts.VAE_train import train
            datamodule, trainer, model, logger, cfg = train(model_name=args.model)

            # test the model
            from scripts.VAE_train import test
            test(datamodule, trainer, model, logger, cfg, save_latent=True)
        elif args.mode == 'build':
            from scripts.VAE_train import train
            train(model_name=args.model, build=True)

    elif args.model == 'LD':
        pass