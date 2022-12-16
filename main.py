from flask import Flask, redirect, request, url_for, render_template
from PIL import Image
from tqdm import tqdm
import torch
import os
import clip
import pathlib
import random

app = Flask(__name__)

model, preprocess = clip.load("ViT-B/32")
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

@app.route('/', methods=("POST", "GET"))
def page():
    if request.method == "POST":
        img_folder = "static/"

        data_dir = pathlib.Path(img_folder)

        images = []
        original_images = []
        nameI = []
        i = 0

        clip.tokenize("Hello world!")

        preprocess

        for filename in tqdm([filename for filename in os.listdir(data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]):
            name = os.path.splitext(filename)[0]
            image = Image.open(os.path.join(data_dir, filename)).convert("RGB")

            nameI.append(name + ".jpg")

            original_images.append(image)
            images.append(preprocess(image))
            i += 1

        descriptions = ["a photograph"]
        descriptions.append(request.form["prompt"])

        text_descriptions = [f"This is a photo of a {label}" for label in descriptions]
        text_tokens = clip.tokenize(text_descriptions)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        image_features = torch.load("/Users/gabrielbp/PycharmProjects/clipWebsite/website/torch.pt")

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(2, dim=-1)

        nameImageTopProb = []
        prob = []

        for i, image in enumerate(original_images):
            if (float(top_probs[i][0]) > 0.98):
                nameImageTopProb.append(nameI[i])
                prob.append(float(top_probs[i][0]))

        return render_template("grid.html", nameI=nameImageTopProb, prob=prob)
    else:
        img_folder = "static/"

        data_dir = pathlib.Path(img_folder)

        nameI = []

        for filename in tqdm([filename for filename in os.listdir(data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]):
            name = os.path.splitext(filename)[0]
            rdm = random.randint(1, 2)
            if rdm == 1:
                nameI = nameI
            else:
                nameI.append(name + ".jpg")

        nameImg = []
        e = 0

        for img in range(200):
            nameImg.append(nameI[e])
            e += 1

        return render_template("home.html", nameI=nameImg)
