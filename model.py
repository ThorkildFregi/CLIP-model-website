import numpy as np
import torch
import os
from PIL import Image
import clip
import pathlib

img_folder = "static/"

print(" reading ", img_folder)
data_dir = pathlib.Path(img_folder)
image_count = len(list(data_dir.glob('*.*')))
print("number of images: ", image_count)

print("Torch version:", torch.__version__)

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters :" f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution :", input_resolution)
print("Context length :", context_length)
print("Vocab size :", vocab_size)

preprocess

clip.tokenize("Hello world!")

original_images = []
images = []
i = 0

for filename in [filename for filename in os.listdir(data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    print(i, " : ", name)
    image = Image.open(os.path.join(data_dir, filename)).convert("RGB")

    original_images.append(image)
    images.append(preprocess(image))
    i += 1

image_input = torch.tensor(np.stack(images))

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)

torch.save(image_features, "torch.pt")
