import torch
import torch.nn as nn


class NormLayer(nn.Module):
    """ Layer that computer embedding normalization """

    def __init__(self, l=2):
        """ Layer initializer """
        assert l in [1, 2]
        super().__init__()
        self.l = l
        return

    def forward(self, x):
        """ Normalizing embeddings x. The shape of x is (B,D) """
        x_normalized = x / torch.norm(x, p=self.l, dim=-1, keepdim=True)
        return x_normalized


class ConvBlock(nn.Module):
    """ Building block with 2 convolutions """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = 'same'
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2)
        )

    def forward(self, x):
        y = self.block(x)
        return y


class EN(nn.Module):
    """
    Implementation of a simple siamese model
    """

    def __init__(self, emb_dim=128, channels=[3, 64, 64, 32]):
        """ Module initializer """
        super().__init__()

        n_layers = len(channels) - 1

        # convolutional feature extractor
        cnn = []
        for i in range(n_layers):
            cnn.append(ConvBlock(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=3))

        self.cnn = nn.Sequential(*cnn)
        self.cnn_last_block = nn.Sequential(
            nn.Conv2d(in_channels=channels[-1], out_channels=channels[-1], kernel_size=3, padding='same'),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU())

        # fully connected embedder
        flat_dim = 64 * 64 * 32
        self.fc = nn.Linear(flat_dim, emb_dim)

        # auxiliar layers
        self.flatten = nn.Flatten()
        self.norm = NormLayer()

        # FC
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.ac1 = nn.ReLU()
        self.do1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.ac2 = nn.ReLU()
        self.do2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 4)
        self.sig = nn.Sigmoid()

    def forward_one(self, x):

        """ Forwarding just one sample through the model """
        x = self.cnn(x)
        x = self.cnn_last_block(x)
        x_flat = self.flatten(x)
        x_emb = self.fc(x_flat)
        x_emb_norm = self.norm(x_emb)
        return x_emb_norm

    def forward(self, x):
        if len(x.size()) == 4:
            x.unsqueeze(0)

        if x.size()[0] == 1:
            x = x.expand(2, 3, 64, 64, 2)

        emb1 = self.forward_one(x.select(4, 0))
        emb2 = self.forward_one(x.select(4, 1))
        emb = torch.cat((emb1, emb2), 1)
        x = self.fc1(emb)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.do2(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x