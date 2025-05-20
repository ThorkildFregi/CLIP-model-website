from flask import Flask, redirect, request, url_for, render_template
from model import InitialiseModel
from tqdm import tqdm
from PIL import Image
import pathlib
import torch
import clip
import os

UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda")

model, preprocess = clip.load("ViT-B/32")
model.to(device)
model.eval()

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

@app.route('/', methods=("POST", "GET"))
def home():
    dataset = request.args.get("dataset", default="gallica_wwi")
    listdir = os.listdir('static/')

    if listdir:
        if request.method == "POST":
            img_folder = f"static/{dataset}/"

            data_dir = pathlib.Path(img_folder)

            images = []
            original_images = []
            nameI = []
            i = 0

            clip.tokenize("Hello world!").to(device)

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
            text_tokens = clip.tokenize(text_descriptions).to(device)

            with torch.no_grad():
                text_features = model.encode_text(text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # load toarch image_features from model.py
            image_features = torch.load(f"{dataset}_tensor.pt").to(device)

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

            return render_template("grid.html", dataset=dataset, nameI=nameImageTopProb, prob=prob)
        else:
            start = f"static/{dataset}"

            for dirpath, dirnames, filenames in os.walk(start):
                if filenames:
                    return render_template("home.html", listdataset=listdir, dataset=dataset, nameI=filenames)
                else:
                    return redirect(url_for("add_dataset"))
    else:
        return redirect(url_for("add_dataset"))

@app.route('/add-dataset', methods=("POST", "GET"))
def add_dataset():
    if request.method == "POST":
        name = request.form["name"]
        model = request.form["model"]

        for folder in os.listdir(UPLOAD_FOLDER):
            if folder == name:
                raise Exception("A folder is already named that way !")
        
        os.makedirs(f"static/{name}/")

        files = request.files.getlist("images")
        for file in files:
            image = Image.open(file)
            image.save(f"static/{name}/{file.filename}")
        InitialiseModel(name, model)
        return redirect(url_for('home'))
    else:
        listmodel = []
        for model in clip.available_models():
            if "ViT" in model:
                listmodel.append(model)

        return render_template("addDataset.html", listmodel=listmodel)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)