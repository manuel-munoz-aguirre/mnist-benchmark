#!/usr/bin/env python
# coding: utf-8

import argparse 
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from six.moves import urllib
from torchvision import datasets, transforms

from STN import SimpleSTN, CoordConvSTN
from spinal import SpinalNet, SpinalNetCoordConv
from vit import ViT
from utils import plot_metric, convert_image_np, compare_stns, plot_wrong_preds
pl.utilities.seed.seed_everything(1) 


def main(args):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=args.bs, shuffle=True, num_workers=args.workers)

    # Validation dataset
    validation_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=args.bs, shuffle=False, num_workers=args.workers)
    
    # Set the model type
    if args.model == "stn":
        model = SimpleSTN(lr = args.lr)
    elif args.model == "stncoordconv":
        model = CoordConvSTN(lr = args.lr, 
                             coordconv_localization=args.localization)
    elif args.model == "vit":
        model_kwargs = {
                "embed_dim": 64,
                "hidden_dim": 128,
                "num_heads": 8,
                "num_layers": 6,
                "patch_size": 7, 
                "num_channels": 1,
                "num_patches": 64,
                "num_classes": 10,
                "dropout": 0.2
                }
        model = ViT(model_kwargs, lr=args.lr)
    elif args.model == "spinal":
        model = SpinalNet(first_HL=8, lr=args.lr)
    elif args.model == "spinalstn":
        model = SpinalNetCoordConv(first_HL=8, lr=args.lr)

    gpus = 1 if torch.cuda.is_available() and args.device == 'gpu' else 0

    logger = TensorBoardLogger("logs", name=args.model)
    early_stop_callback = EarlyStopping(monitor="validation_loss", 
                                        min_delta=args.mindelta,
                                        patience=args.patience, 
                                        verbose=True, 
                                        mode="min")
    
    trainer = pl.Trainer(logger=logger,
                         callbacks=[early_stop_callback],
                         gpus=gpus,
                         min_epochs=1,
                         max_epochs=args.maxepochs,
                         deterministic=True)
    trainer.fit(model,
                train_dataloaders=train_loader, 
                val_dataloaders=validation_loader)

    print("Validation accuracy = %.04f and loss = %.04f at epoch %d" %
      (trainer.logged_metrics['validation_acc'], 
       trainer.logged_metrics['validation_loss'], 
       trainer.logged_metrics['epoch']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'MNIST-benchmarks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='cpu', 
                        help='Device on which to run the experiments.')
    parser.add_argument('--workers', type=int, default=2, 
                        help='Number of workers for dataloaders.')
    parser.add_argument('--bs', type=int, default=64, 
                        help='Batch size.')
    parser.add_argument('--maxepochs', type=int, metavar='MAX_EPOCHS', default=20, 
                        help='Maximum number of epochs to run the experiment for.')
    parser.add_argument('--patience', type=int, metavar='PATIENCE', default=5, 
                        help='Number of epochs with no improvement before triggering early stopping.')
    parser.add_argument('--mindelta', type=float, metavar='MIN_DELTA', default=0.005, 
                        help='Required improvement in the validation loss for early stopping.')
    parser.add_argument('--model', type=str, choices=['stn', 'stncoordconv', 'vit', 'spinal', 'spinalstn'], default='stn', help='Type of model to train.')
    parser.add_argument('--localization', default=False, action='store_true', 
                        help='Whether to use CoordConv in the localization network.')
    parser.add_argument('--lr', type=float, metavar='LR', default=0.01, 
                        help='Learning rate for SGD.')
    parser.add_argument('--logs', type=str, metavar='LOGPATH', default='logs/', 
                        help='Directory to store tensorboard logs.')

    args = parser.parse_args()
    
    main(args)