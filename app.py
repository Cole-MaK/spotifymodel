import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=["GET","POST"])
def predict_genre():
    if request == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            song_name=request.form.get('song_name'),
            artist=request.form.get('artist')
        )
        song_data = data.get_song_data()

        pred_pipeline = PredictPipeline()
        result = pred_pipeline.predict(song_data)
        result = result.astype(int)[0]
        genre_dict = load_object('artifact/genre_dict.pkl')
        return render_template('home.html', results = genre_dict[result])