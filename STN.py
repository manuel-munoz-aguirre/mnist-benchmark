import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F

from modules import CoordConv

class SimpleSTN(pl.LightningModule):
    """
    Pytorch-lightning implementation of a simple spatial
    transformer network, based on the official pytorch tutorial at:
    https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    """

    def __init__(self, lr):
        """
        Args:
          lr: Learning rate for SGD.
        """

        super(SimpleSTN, self).__init__()

        # Hyperparameters and metrics
        self.save_hyperparameters()
        self.lr = lr
        self.train_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()
        self.example_input_array = torch.rand((64, 1, 28, 28))

        # Model parameters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0],
                                                    dtype=torch.float))


    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x


    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        return optimizer


    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = F.nll_loss(logits, target)
        self.train_accuracy(logits, target)

        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('train_acc', self.train_accuracy, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        data, target = batch

        logits = self(data)
        loss = F.nll_loss(logits, target)
        self.validation_accuracy(logits, target)

        self.log('validation_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('validation_acc', self.validation_accuracy, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss


class CoordConvSTN(pl.LightningModule):
    """
    Pytorch-lightning implementation of a STN with
    convolutional layers replaced by CoordConv layers.
    """

    def __init__(self, lr, coordconv_localization=False, with_r=False):
        """
        Args:
          lr: Learning rate for SGD.
        """

        super(CoordConvSTN, self).__init__()

        # Hyperparameters and metrics
        self.save_hyperparameters()
        self.lr = lr
        self.coordconv_localization = coordconv_localization
        self.with_r = with_r
        self.train_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()
        self.example_input_array = torch.rand((64, 1, 28, 28))

        # Model parameters
        self.coordconv1 = CoordConv(x_dim=28, y_dim=28, with_r=self.with_r,
                                    in_channels=1, out_channels=10,
                                    kernel_size=5)
        self.coordconv2 = CoordConv(x_dim=12, y_dim=12, with_r=self.with_r,
                                    in_channels=10, out_channels=20,
                                    kernel_size=5)
        self.coordconv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        if self.coordconv_localization:
            self.localization = nn.Sequential(
              CoordConv(x_dim=28, y_dim=28, with_r=self.with_r,
                        in_channels=1, out_channels=8, kernel_size=7),
              nn.MaxPool2d(2, stride=2),
              nn.ReLU(True),
              CoordConv(x_dim=11, y_dim=11, with_r=self.with_r,
                        in_channels=8, out_channels=10, kernel_size=5),
              nn.MaxPool2d(2, stride=2),
              nn.ReLU(True)
            )
        else:
            self.localization = nn.Sequential(
              nn.Conv2d(1, 8, kernel_size=7),       # (bs, 8, 22, 22)
              nn.MaxPool2d(2, stride=2),            # (bs, 8, 11, 11)
              nn.ReLU(True),                        # (bs, 8, 11, 11)
              nn.Conv2d(8, 10, kernel_size=5),      # (bs, 10, 7, 7)
              nn.MaxPool2d(2, stride=2),            # (bs, 10, 3, 3)
              nn.ReLU(True)
          )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0],
                                                    dtype=torch.float))


    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x


    def forward(self, x):
        x = self.stn(x)                            # (bs, 1, 28, 28)
        x = self.coordconv1(x)                     # (bs, 10, 24, 24)
        x = F.max_pool2d(x, 2)                     # (bs, 10, 12, 12)
        x = F.relu(x)                              # (bs, 10, 12, 12)
        x = self.coordconv2(x)                     # (bs, 20, 8, 8)
        x = self.coordconv2_drop(x)                # (bs, 20, 8, 8)
        x = F.max_pool2d(x, 2)                     # (bs, 20, 4, 4)
        x = F.relu(x)                              # (bs, 20, 4, 4)
        x = x.view(-1, 320)                        # (bs, 320)
        x = self.fc1(x)                            # (bs, 50)
        x = F.relu(x)                              # (bs, 50)
        x = F.dropout(x, training=self.training)   # (bs, 50)
        x = self.fc2(x)                            # (bs, 10)

        return F.log_softmax(x, dim=1)


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        return optimizer


    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = F.nll_loss(logits, target)
        self.train_accuracy(logits, target)

        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('train_acc', self.train_accuracy, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = F.nll_loss(logits, target)
        self.validation_accuracy(logits, target)

        self.log('validation_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('validation_acc', self.validation_accuracy, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss
