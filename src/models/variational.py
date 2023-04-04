import torch
import torch.nn as nn
import torch.nn.functional as F   


class VariationalAutoEncoder(nn.Module):

    def _relative_projection(self, x, anchors):
        # Perform relative projection
        x = F.normalize(x, dim=1)
        anchors = F.normalize(anchors, dim=1)
        return torch.einsum("im,jm -> ij", x, anchors)

    def _new_encoded_anchors(self):
        # no grad
        with torch.no_grad():
            # Compute new encoded anchors
            mean, logvar = self.encode(self.anchors)
            new_anchors = self._reparameterize(mean, logvar)
        return new_anchors
    
    def _reparameterize(self, mean, logvar):
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)
        else:
            return mean

    def __init__(self, anchors=None, layer_size = 32, use_relative_space=False, hidden_size=10):
        super(VariationalAutoEncoder, self).__init__()
        self.anchors = anchors
        self.latent_dim = hidden_size
        self.use_relative_space = use_relative_space
        self.max_logvar = 5 # max log variance, to avoid numerical instability

        if self.use_relative_space:
            # move the anchors to the GPU
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.anchors = self.anchors.to(device)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, layer_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(layer_size, layer_size*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * layer_size*2, 2*self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 7 * 7 * layer_size),
            nn.ReLU(),
            nn.Unflatten(1, (layer_size, 7, 7)),
            nn.ConvTranspose2d(layer_size, layer_size*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(layer_size*2, layer_size, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(layer_size, 1, kernel_size=3, stride=1, padding=1),
        )

    def encode(self, x):
        """Encode a batch of images."""
        mean, logvar = torch.chunk(self.encoder(x), 2, dim=1)
        logvar = torch.clamp(logvar, min=-self.max_logvar, max=self.max_logvar)
        return mean, logvar

    def decode(self, z):
        """Decode a batch of latent vectors."""
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self._reparameterize(mean, logvar)
        if self.use_relative_space:
            new_anchors = self._new_encoded_anchors()
            z = self._relative_projection(z, new_anchors)
        return self.decode(z), mean, logvar

    def loss_function(self, recon_x, x, mean, logvar):
        """Compute the loss function for the VAE."""
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return BCE + KLD







