from flask import Flask, redirect, request, url_for, render_template
from model import InitialiseModel
from PIL import Image
from tqdm import tqdm
import pathlib
import shutil
import torch
import clip
import os

UPLOAD_FOLDER = '/static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model, preprocess = clip.load("ViT-B/32")
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=("POST", "GET"))
def home():
    listdir = os.listdir('static/')

    if listdir:
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
                image = Image.open(os.path.join(data_dir, filename)).convert("RGB")

                nameI.append(filename)

                original_images.append(image)
                images.append(preprocess(image))
                i += 1

            # description from user
            descriptions = ["a photograph"]
            descriptions.append(request.form["prompt"])

            text_descriptions = [f"This is a photo of a {label}" for label in descriptions]
            text_tokens = clip.tokenize(text_descriptions)

            with torch.no_grad():
                text_features = model.encode_text(text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # load toarch image_features from model.py
            image_features = torch.load("tensor.pt")

            # top probability
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_probs, top_labels = text_probs.cpu().topk(2, dim=-1)

            nameImageTopProb = []
            prob = []

            # top prob
            for i, image in enumerate(original_images):
                if (float(top_probs[i][0]) > 0.98):
                    nameImageTopProb.append(nameI[i])
                    prob.append(float(top_probs[i][0]))

            print(prob)

            return render_template("grid.html", nameI=nameImageTopProb, prob=prob)
        else:
            start = "static/"

            for dirpath, dirnames, filenames in os.walk(start):
                if filenames:
                    return render_template("home.html", nameI=filenames)  # nameI=nameImg)
                else:
                    return redirect(url_for("changeImages"))
    else:
        return redirect(url_for("changeImages"))

@app.route('/change-image', methods=("POST", "GET"))
def changeImages():
    if request.method == "POST":
        folder = 'static/'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        files = request.files.getlist("images")
        for file in files:
            image = Image.open(file)
            image.save(f"static/{file.filename}")
        InitialiseModel()
        return redirect(url_for('home'))
    else:
        return render_template("changeImages.html")
