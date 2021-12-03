import torch
from torchmetrics.functional import accuracy
import torch.nn as nn
import torch.nn.functional as TF
import torch.optim as optim
# Ignore excessive warnings
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)


from pytorch_lightning import LightningModule


class IcmeNet(LightningModule):
    def __init__(
            self,
            num_classes,
            train_loader,
            test_loader,
            val_loader,
            kernel_size=3
    ):
        """CNN used to classify ICMEs

        Parameters
        ----------
        num_classes : int
            Number of classes, will always be two for us
        train_loader : pytorch.utils.data.Dataset
            Pytorch dataloader for training dataset
        test_loader : pytorch.utils.data.Dataset
            Pytorch dataloader for testing dataset
        val_loader : pytorch.utils.data.Dataset
            Pytorch dataloader for validation set
        kernel_size : int
            Kernel to use in convolution layers
        """
        super(IcmeNet, self).__init__()

        self.num_classes = num_classes
        self._train_dataloader = train_loader
        self._test_dataloder = test_loader
        self._val_dataloader = val_loader
        self._input_dim = train_loader.dataset[0][0].shape[1]
        self.kernel_size = kernel_size

        # Defining the network layers
        self.conv1 = nn.Conv2d(8, 16, kernel_size=self.kernel_size, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.kernel_size, stride=1)
        self.dropout = nn.Dropout(0.10)
        self.fc1 = nn.Linear(32 * 24 * 24, 8192)
        self.fc2 = nn.Linear(8192, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x = x.float()
        x = TF.relu(TF.max_pool2d(self.conv1(x), 4))
        x = TF.relu(TF.max_pool2d(self.conv2(x), 4))
        x = x.view(-1, 32 * 24 * 24)
        x = TF.relu(self.fc1(x))
        x = TF.relu(self.fc2(x))
        x = TF.relu(self.fc3(x))
        x = TF.relu(self.fc4(x))
        # No activation on output layer because we are using cross entropy loss
        x = self.fc5(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, = batch
        y_hat = self(x)
        loss = TF.cross_entropy(y_hat, y)
        self.log("Train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y, = batch
        y_hat = self(x)
        loss = TF.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self, LR=0.001):
        optimizer = optim.Adam(self.parameters(), lr=LR)
        return {"optimizer": optimizer}

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

