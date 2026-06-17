import torch
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms

from models import Generator
from config import *

# --------------------------------
# Device
# --------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------
# Dataset (для списка классов)
# --------------------------------

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

dataset = ImageFolder(dataset_path, transform=transform)
classes = dataset.classes

# --------------------------------
# SHOW CLASSES
# --------------------------------

print("Материалы:")
for i, c in enumerate(classes):
    print(f"{i} - {c}")

choice = int(input("Выберите материал: "))
material_name = classes[choice]

# --------------------------------
# LOAD MODEL (ВАЖНО: по материалу)
# --------------------------------

model_path = f"models_saved/{material_name}_generator.pth"

G = Generator(1).to(device)

G.load_state_dict(
    torch.load(model_path, map_location=device)
)

G.eval()

# --------------------------------
# GENERATE
# --------------------------------

noise = torch.randn(
    1,
    latent_dim,
    device=device
)

label = torch.zeros(
    1,
    dtype=torch.long,
    device=device
)

with torch.no_grad():
    generated = G(noise, label)

# --------------------------------
# SAVE
# --------------------------------

output_path = f"outputs/{material_name}.png"

save_image(
    generated,
    output_path,
    normalize=True
)

print(f"Сохранено: {output_path}")