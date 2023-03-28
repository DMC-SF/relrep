import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import load_anchors


class AutoEncoder(nn.Module):

    def __init__(self, layer_size = 16, hidden_size=10, use_relative_space=False):
        super().__init__()

        # encoder with pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(1, layer_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(layer_size),
            nn.ReLU(),
            nn.Conv2d(layer_size, layer_size*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(layer_size*2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * layer_size*2, hidden_size),
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

        self.use_relative_space = use_relative_space
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.anchors = load_anchors(data_dir="data", anchors_dir="anchors").to(device) # (N_ANC,1,28,28)

    def _relative_projection(self, x, anchors):
        # (B,N_ANC)@(N_ANC,N_ANC) -> (B,N_ANC)
        x = F.normalize(x, dim=1)
        anchors = F.normalize(anchors, dim=1)
        return torch.einsum("im,jm -> ij", x, anchors)
    
    def _new_encoded_anchors(self):
        # with torch.no_grad():
        out = self.encoder(self.anchors)
        return out

    def forward(self, x):
        encoded = self.encoder(x)
        if self.use_relative_space:
            new_anchors = self._new_encoded_anchors() # TODO: choose if we wanto the no_grad or not
            encoded = self._relative_projection(encoded, new_anchors) 
        decoded = self.decoder(encoded)
        return decoded