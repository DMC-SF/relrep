import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

class ConvAutoencoder(pl.LightningModule):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 2)
        )
        
        # Fixed points
        self.fixed_points = torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype=torch.float32)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 7 * 7 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.euclidean_distance_layer(x)
        x = self.decoder(x)
        return x

    def euclidean_distance_layer(self, x):
        distances = torch.cdist(x, self.fixed_points.to(x.device))
        return distances

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        img, _ = batch
        outputs = self(img)
        loss = nn.MSELoss()(outputs, img)
        self.log('train_loss', loss)
        return loss
class ConvAutoencoder(pl.LightningModule):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 2)
        )
        
        # Fixed points
        self.fixed_points = torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype=torch.float32)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 7 * 7 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.euclidean_distance_layer(x)
        x = self.decoder(x)
        return x

    def euclidean_distance_layer(self, x):
        distances = torch.cdist(x, self.fixed_points.to(x.device))
        return distances

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        img, _ = batch
        outputs = self(img)
        loss = nn.MSELoss()(outputs, img)
        self.log('train_loss', loss)
        return loss