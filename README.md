# CLIP model website

A website combining CLIP, Masonry, Flask and UIkit. I use old image from Gallica during the WWI.

To use this code you will need :
- Flask : ```pip install flask```
- CLIP : ```pip install git+https://github.com/openai/CLIP.git```

You have two python files :
- ```model.py``` : where you can find all the code to take the image features.
- ```main.py``` : where you can find all the code of the website

In the folder ```templates```, you can find all the HTML files :
- ```home.html``` : where you can find all the code for the home page with the first Masonry grid and the prompt.
- ```grid.html``` : where you can find all the code for the result page with the second Masonry grid and the probability of all the pictures.

In the folder ```static```, you can find all the images of Gallica.

Before launching the website, run the ```model.py``` you will view a ```torch.pt``` file appear in your folder. This files contain ```image_features```. We are going to use this file in ```main.py```.

After, verify you are in the same folder as ```main.py``` in your terminal and type : ```flask --app main.py run``` or ```flask --app main.py --debug run``` if you want the debug mode.

Then go on ```https://127.0.0.1/``` and have fun !

You can use the code for your proper images in changing the images in the folder ```static``` and rerun the ```model.py```.

Thanks to Jean-Philippe Moreux for the help !
