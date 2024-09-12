import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

song = str(input('song name:'))
artist = str(input('artist:'))
data = CustomData(
    song_name=song,
    artist=artist
)
song_data = data.get_song_data()

pred_pipeline = PredictPipeline()
result = pred_pipeline.predict(song_data)