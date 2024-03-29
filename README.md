# CLIP model website

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Presentation**

A website combining CLIP, Masonry, Flask and UIkit using old image from Gallica during the WWI.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Installation**

To use this code you will need :
- Flask : ```pip install flask```
- CLIP : ```pip install git+https://github.com/openai/CLIP.git```

You have two python files :
- ```model.py``` : where you can find all the code to take the image features.
- ```main.py``` : where you can find all the code of the website

In the folder ```templates```, you can find all the HTML files :
- ```home.html``` : where you can find all the code for the home page with the first Masonry grid and the prompt.
- ```grid.html``` : where you can find all the code for the result page with the second Masonry grid and the probability of all the pictures.
- ```changeImages.html``` : where you can find all the code for the page to change the pictures.

In the folder ```static```, you can find all the images of Gallica.

After, verify you are in the same folder as ```main.py``` in your terminal and type : ```flask --app main.py run``` or ```flask --app main.py --debug run``` if you want the debug mode.

Then go on ```http://127.0.0.1/``` and have fun !

Before asking the AI, initialise the model.

You can use the code for your proper images in changing the images in the folder ```static``` and rerun the ```model.py``` or change the image with the website page but attention you can't do it with a lot of image.

Thanks to Jean-Philippe Moreux for the help !

Remix version by Jean-Philippe Moreux : https://github.com/altomator/CLIP_test/

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
