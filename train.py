import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os

from models import Generator, Discriminator
from config import *


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

class TextureDataset(Dataset):

    def __init__(self, folder, transform=None):

        self.folder = folder
        self.transform = transform

        self.images = []

        for file in os.listdir(folder):

            if file.endswith((".png", ".jpg", ".jpeg")):

                self.images.append(
                    os.path.join(folder, file)
                )


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        img = Image.open(
            self.images[index]
        ).convert("L")


        if self.transform:
            img = self.transform(img)

        label = 0


        return img, label

# =========================
# МАТЕРИАЛЫ
# =========================

materials = [
    "brick",
    "wood",
]


# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([

    transforms.Resize(
        (image_size, image_size)
    ),

    transforms.Grayscale(
        num_output_channels=1
    ),
    
    transforms.GaussianBlur(3),
    
    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )

])


# =========================
# ОБУЧЕНИЕ КАЖДОГО КЛАССА
# =========================

for material in materials:


    print("\n================")
    print(
        "Обучение:",
        material
    )
    print("================")


    dataset = TextureDataset(
    f"{dataset_path}/{material}",
    transform
)


    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=True,

        drop_last=True

    )


    # один класс
    num_classes = 1



    G = Generator(
        num_classes
    ).to(device)
   
    D = Discriminator(
        num_classes
    ).to(device)
   
    criterion = nn.BCELoss()
    
    optimizer_G = optim.Adam(
        G.parameters(),
        lr=learning_rate,
        betas=(0.5,0.999)
    )

    optimizer_D = optim.Adam(
        D.parameters(),
        lr=learning_rate,
        betas=(0.5,0.999)
    )



    # =========================
    # EPOCHS
    # =========================
    for epoch in range(epochs):
        for imgs,_ in loader:
            imgs = imgs.to(device)
            bs = imgs.size(0)
            real = torch.ones(bs, 1, device=device)
            real = real * (0.8 + 0.2 * torch.rand(bs, 1, device=device))  # 0.8–1.0
            fake = torch.zeros(bs, 1, device=device)
            fake = fake + (0.1 * torch.rand(bs, 1, device=device))        # 0.0–0.1
            
            # =====================
            # DISCRIMINATOR
            # =====================
            optimizer_D.zero_grad()
            output = D(
                imgs,
                torch.zeros(bs,
                dtype=torch.long,
                device=device)
            )
            loss_real = criterion(
                output,
                real
            )
            noise = torch.randn(
                bs,
                latent_dim,
                device=device
            )
            fake_imgs = G(
                noise,
                torch.zeros(
                    bs,
                    dtype=torch.long,
                    device=device
                )
            )
            output_fake = D(
                fake_imgs.detach(),
                torch.zeros(
                    bs,
                    dtype=torch.long,
                    device=device
                )
            )
            loss_fake = criterion(
                output_fake,
                fake
            )
            d_loss = loss_real + loss_fake
            d_loss.backward()
            optimizer_D.step()
            # =====================
            # GENERATOR
            # =====================
            optimizer_G.zero_grad()
            output = D(
                fake_imgs,
                torch.zeros(
                    bs,
                    dtype=torch.long,
                    device=device
                )
            )
            g_loss = criterion(
                output,
                real
            )
            g_loss.backward()
            optimizer_G.step()
        print(
            f"{material} | "
            f"Epoch {epoch+1}/{epochs} "
            f"D:{d_loss.item():.4f} "
            f"G:{g_loss.item():.4f}"
        )
    # =========================
    # SAVE
    # =========================
    torch.save(
        G.state_dict(),
        f"models_saved/{material}_generator.pth"
    )
    print(
        material,
        "готов"
    )
print("Все материалы обучены")