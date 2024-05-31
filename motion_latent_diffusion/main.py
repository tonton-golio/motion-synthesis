
import argparse
from utils import load_config


if __name__ == "__main__":
    # add arguments for model and mode
    parser = argparse.ArgumentParser(description='Run the model')
    parser.add_argument('--model_name', type=str, default='VAE', help='Model to run')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run')
    parser.add_argument('--data', type=str, default='motion', help='Data to run')
    args = parser.parse_args()
    print(args.model_name, args.mode)
    assert args.model_name.upper() in ['VAE1', 'VAE4', 'VAE5',  # MotionVAE
                          'LD_VAE1', 'LD_VAE4', 'LD_VAE5', # Latent Diffusion
                            'LINEAR', 'GRAPH', 'CONV'  # PoseVAE
                          ]
    assert args.mode.lower() in ['train', 'build', 'inference', 'optuna']

    assert args.data.lower() in ['motion', 'pose']


    if args.data == 'motion':
        
        if args.model_name[:3] == 'VAE':
            if args.mode == 'train':
                # train the model
                from scripts.MotionVAE_train import train
                datamodule, trainer, model, logger, cfg = train(model_name=args.model_name)

                # test the model
                from scripts.MotionVAE_train import test
                test(datamodule, trainer, model, logger, cfg, save_latent=True)
            elif args.mode == 'build':
                from scripts.MotionVAE_train import train
                train(model_name=args.model_name, build=True)

        elif args.model_name[:2] == 'LD':
            if args.mode == 'train':
                from scripts.MotionLD_train import train
                train(VAE_version=args.model_name.split('_')[1])
            elif args.mode == 'inference':
                from scripts.MotionLD_train import inference
                inference(VAE_version=args.model_name.split('_')[1])

    elif args.data == 'pose':
        assert args.model_name in ['LINEAR', 'GRAPH', 'CONV']
        from scripts.PoseVAE_train import train_test_poseVAE
        train_test_poseVAE(model_type=args.model_name.upper())

