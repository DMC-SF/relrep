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
            out = self.encoder(self.anchors)
        return out

    def __init__(self, anchors, layer_size = 32, use_relative_space=False, hidden_size=500):
        super().__init__()

        self.use_relative_space = use_relative_space

        # Move anchors to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.anchors = anchors.to(device) # (N_ANC,1,28,28)

        # Set hidden size
        self.hidden_size = self.anchors.shape[0] if use_relative_space else hidden_size

        # encoder and decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, layer_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(layer_size, layer_size*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * layer_size*2, self.latent_dim),
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


        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, layer_size, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(layer_size, layer_size*2, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(7 * 7 * layer_size*2, hidden_size),
        # )

        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_size, 7 * 7 * layer_size*2),
        #     nn.ReLU(),
        #     nn.Unflatten(1, (layer_size*2, 7, 7)),
        #     nn.ConvTranspose2d(layer_size*2, layer_size, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(layer_size, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.Sigmoid(),
        # )

    def forward(self, x):
        encoded = self.encoder(x)

        # Use relative space if specified
        if self.use_relative_space:
            new_anchors = self._new_encoded_anchors() # TODO: choose if we wanto the no_grad or not
            encoded = self._relative_projection(encoded, new_anchors) 
    
        decoded = self.decoder(encoded)
        return decoded