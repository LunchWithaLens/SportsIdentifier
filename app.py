from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
from pathlib import Path



# Import fast.ai Library
from fastai import *
from fastai.vision import *

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()


path = Path("path")
classes =  ['aussie','baseball','cricket','football','gaelic','hockey','lacrosse','other','rugby','soccer']
sports = ['Australian Rules Football', 'Baseball', 'Cricket', 'American Football', 'Gaelic Football', 'Field Hockey', 'Lacrosse', 'not a sport','Rugby', 'Soccer']
data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=448).normalize(imagenet_stats)
learn = create_cnn(data2, models.resnet34)
learn.load('LargeModel10Own', strict=False, with_opt=False)




def model_predict(img_path):
    """
       model_predict will return the preprocessed image
    """
   
    img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    pred_percent = (int(100 * (outputs[pred_idx])))
    second = sorted( [(x,i) for (i,x) in enumerate(outputs)], reverse=True )[:2] 
    second_sport = (sports[(second[1][1])])
    second_percent = (int(100 * (outputs[(second[1][1])])))
    return (str(pred_class), str(pred_percent), str(second_sport), str(second_percent))
    




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds, preds_percent, second_sport, second_percent = model_predict(file_path)
        try:
            os.remove(file_path)
        except OSError:
            pass
        sport = sports[classes.index(preds)] + ' and my confidence is ' + preds_percent + '%\n' 
        if int(preds_percent) < 98:
            sport = sport + 'but if that is wrong a second guess would be ' + second_sport + '.'
        return sport
    return None


if __name__ == '__main__':
    
    app.run()


