import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def _relative_projection(self, x, anchors):
        # Perform relative projection
        x = F.normalize(x, dim=1)
        anchors = F.normalize(anchors, dim=1)
        return torch.einsum("im,jm -> ij", x, anchors)
    

    def _new_encoded_anchors(self):
        # no grad
        with torch.no_grad():
            # Compute new encoded anchors
            new_anchors = self.encoder(self.anchors)
            # reparameterize
            mean, logvar = torch.chunk(new_anchors, 2, dim=1)
            new_anchors = self.reparameterize(mean, logvar)
        return new_anchors

    def _reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def __init__(self, anchors, layer_size = 16, use_relative_space=False, hidden_size=10):
        super().__init__()

        self.use_relative_space = use_relative_space
        self.eps = 1e-6
        self.MAX_LOGSTD = 10

        # Move anchors to device
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.anchors = anchors.to(device) # (N_ANC,1,28,28)

        # Set hidden size
        self.hidden_size = self.anchors.shape[0] if use_relative_space else hidden_size

        # encoder and decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, layer_size, kernel_size=3, stride=2, padding=1),
            #nn.InstanceNorm2d(layer_size),
            nn.ReLU(),
            nn.Conv2d(layer_size, layer_size*2, kernel_size=3, stride=2, padding=1),
            #nn.InstanceNorm2d(layer_size*2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * layer_size*2, 2*hidden_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 7 * 7 * layer_size*2),
            nn.ReLU(),
            nn.Unflatten(1, (layer_size*2, 7, 7)),
            nn.ConvTranspose2d(layer_size*2, layer_size, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(layer_size, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, data):
        """
        :param data: data
        :return: mu, logvar
        """
        mu, logvar = self.encoder(data)
        logvar = logvar.clamp(max=self.MAX_LOGSTD) # we need it to avoid inf loss
        return mu, logvar
    
    def reparameterize(self, mu, logstd):
        """
        torch.randn_like(input) -> Returns a tensor with the same size as input that is filled 
        with random numbers from a normal distribution with mean 0 and variance 1
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return: (Tensor)
        """
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, x):
        encoded = self.encoder(x)

        # Reparameterize
        mean, logvar = torch.chunk(encoded, 2, dim=1)
        z = self.reparameterize(mean, logvar)

        # Use relative space if specified
        if self.use_relative_space:
            new_anchors = self._new_encoded_anchors() # TODO: choose if we wanto the no_grad or not
            encoded = self._relative_projection(z, new_anchors) 
    
        decoded = self.decoder(z)
        return decoded, mean, logvar