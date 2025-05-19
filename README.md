# CLIP model website

## Presentation

A website combining CLIP, Masonry, Flask and UIkit using old image from Gallica during the WWI.

## Installation

*Step 1 :*

Clone repositeries.

```git clone https://github.com/ThorkildFregi/CLIP-model-website```

*Step 2 :*

Install dependencies.

```pip install -r requirements.txt```

*Step 3 :*

Run main.py

## Content

You have two python files :
- ```model.py``` : where you can find all the code to take the image features.
- ```main.py``` : where you can find all the code of the website

In the folder ```templates```, you can find all the HTML files :
- ```home.html``` : where you can find all the code for the home page with the first Masonry grid and the prompt.
- ```grid.html``` : where you can find all the code for the result page with the second Masonry grid and the probability of all the pictures.
- ```changeImages.html``` : where you can find all the code for the page to change the pictures.

In the folder ```static```, you can find all the images of Gallica.

## Usage

Go to [http://0.0.0.0:7860](http://0.0.0.0:7860)

Before asking the AI, initialise the model.

You can use the code for your proper images in changing the images in the folder ```static``` and rerun the ```model.py``` or change the image with the website page but attention you can't do it with a lot of image.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Thanks to Jean-Philippe Moreux for the help !

Remix version by Jean-Philippe Moreux : https://github.com/altomator/CLIP_test/

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
