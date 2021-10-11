import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from modules import CoordConv

class SpinalNet(pl.LightningModule):
    """
    Pytorch-lightning port of SpinalNets by Kabir et al.
    (2020, arXiv:2007.03347) Based on the source
    code at: https://github.com/dipuk0506/SpinalNet
    """

    def __init__(self, first_HL, lr):
        """
        Args:
          first_HL: Neurons per FC layer.

          lr: Learning rate for SGD.
        """
        super(SpinalNet, self).__init__()

        # Hyperparameters and metrics
        self.save_hyperparameters()
        self.first_HL = first_HL
        self.lr = lr
        self.train_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()

        # Model parameters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, self.first_HL)
        self.fc1_1 = nn.Linear(160 + self.first_HL, self.first_HL)
        self.fc1_2 = nn.Linear(160 + self.first_HL, self.first_HL)
        self.fc1_3 = nn.Linear(160 + self.first_HL, self.first_HL)
        self.fc1_4 = nn.Linear(160 + self.first_HL, self.first_HL)
        self.fc1_5 = nn.Linear(160 + self.first_HL, self.first_HL)
        self.fc2 = nn.Linear(self.first_HL*6, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x1 = x[:, 0:160]

        x1 = F.relu(self.fc1(x1))
        x2= torch.cat([ x[:,160:320], x1], dim=1)
        x2 = F.relu(self.fc1_1(x2))
        x3= torch.cat([ x[:,0:160], x2], dim=1)
        x3 = F.relu(self.fc1_2(x3))
        x4= torch.cat([ x[:,160:320], x3], dim=1)
        x4 = F.relu(self.fc1_3(x4))
        x5= torch.cat([ x[:,0:160], x4], dim=1)
        x5 = F.relu(self.fc1_4(x5))
        x6= torch.cat([ x[:,160:320], x5], dim=1)
        x6 = F.relu(self.fc1_5(x6))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = torch.cat([x, x5], dim=1)
        x = torch.cat([x, x6], dim=1)

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


class SpinalNetCoordConv(pl.LightningModule):
    """
    Spinal net with the convolutional layers replaced by
    CoordConv layers, with an STN transformation.
    """

    def __init__(self, first_HL, lr, with_r=False):
        """
        Args:
          first_HL: Neurons per FC layer.

          lr: Learning rate for SGD.

          with_r: Whether to add an r-coordinate channel.
        """
        super(SpinalNetCoordConv, self).__init__()

        # Hyperparameters and metrics
        self.save_hyperparameters()
        self.first_HL = first_HL
        self.lr = lr
        self.with_r = with_r
        self.train_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()

        # Model parameters
        self.conv1 = CoordConv(x_dim=28, y_dim=28, with_r=self.with_r,
                               in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = CoordConv(x_dim=12, y_dim=12, with_r=self.with_r,
                               in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, self.first_HL)
        self.fc1_1 = nn.Linear(160 + self.first_HL, self.first_HL)
        self.fc1_2 = nn.Linear(160 + self.first_HL, self.first_HL)
        self.fc1_3 = nn.Linear(160 + self.first_HL, self.first_HL)
        self.fc1_4 = nn.Linear(160 + self.first_HL, self.first_HL)
        self.fc1_5 = nn.Linear(160 + self.first_HL, self.first_HL)
        self.fc2 = nn.Linear(self.first_HL*6, 10)

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
        # Transform the input
        x = self.stn(x)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x1 = x[:, 0:160]

        x1 = F.relu(self.fc1(x1))
        x2= torch.cat([ x[:,160:320], x1], dim=1)
        x2 = F.relu(self.fc1_1(x2))
        x3= torch.cat([ x[:,0:160], x2], dim=1)
        x3 = F.relu(self.fc1_2(x3))
        x4= torch.cat([ x[:,160:320], x3], dim=1)
        x4 = F.relu(self.fc1_3(x4))
        x5= torch.cat([ x[:,0:160], x4], dim=1)
        x5 = F.relu(self.fc1_4(x5))
        x6= torch.cat([ x[:,160:320], x5], dim=1)
        x6 = F.relu(self.fc1_5(x6))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = torch.cat([x, x5], dim=1)
        x = torch.cat([x, x6], dim=1)

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
