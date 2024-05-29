
from pytorch_lightning import Trainer
import torch
from utils import plotUMAP, prep_save, save_for_diffusion


def train(dm , criteria, config, logger, MODEL):
    # instantiate model
    model = MODEL(criteria, **config['TRAIN']['MODEL'])
    
    # train
    trainer = Trainer(logger=logger, **config['TRAIN']['TRAINER'])
    print(logger.log_dir)
    log_dir_base = logger.log_dir.split('/version')[0]
    trainer.fit(model, dm,
                #ckpt_path = log_dir_base+'/version_38/checkpoints/epoch=19-step=15000.ckpt'
                )

    return model, trainer

    

def test(dm , trainer, model, logger, config, save_latent=False):
    # test
    trainer.test(model, datamodule=dm)
    res = model.on_test_epoch_end()
    logger.log_hyperparams(model.hparams, {'MSE (test, unscaled)' : res[0]} )

    # save_latent
    if save_latent:
        dataloaders = [dm.test_dataloader(), dm.train_dataloader(), dm.val_dataloader()]
        KL_weight = config['TRAIN']['LOSS']['DIVERGENCE_KL']
        latent, labels = prep_save(model, dataloaders, enable_y=False, log_dir=logger.log_dir)
        latent_dim = torch.prod(torch.tensor(latent.shape[1:]))
        latent = latent.view(-1, latent_dim)

        projection, reducer = plotUMAP(latent, labels, latent_dim, KL_weight, logger.log_dir, show=False, max_points=5000)
        
        save_for_diffusion(save_path=logger.log_dir+'/saved_latent', model = model, z = latent, y = labels, projection = projection, projector = reducer,  )
