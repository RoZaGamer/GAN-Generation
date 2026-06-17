import torch
import torch.nn as nn
from config import image_size, latent_dim


# =========================
# GENERATOR
# =========================
class Generator(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(

            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.ReLU(True),

            nn.Linear(512, 1024),
            nn.ReLU(True),

            nn.Linear(1024, image_size * image_size),
            nn.Tanh()
        )

        self.img_size = image_size

    def forward(self, noise, labels):

        label_emb = self.label_emb(labels)

        x = torch.cat([noise, label_emb], dim=1)

        x = self.model(x)

        x = x.view(x.size(0), 1, self.img_size, self.img_size)

        return x


# =========================
# DISCRIMINATOR
# =========================
class Discriminator(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, image_size * image_size)

        self.model = nn.Sequential(

            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):

        batch_size = img.size(0)

        label = self.label_emb(labels)

        label = label.view(
            batch_size,
            1,
            image_size,
            image_size
        )

        x = torch.cat([img, label], dim=1)

        x = self.model(x)

        return x.view(batch_size, 1)