import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, get_token, get_auth_header
from src.components.data_transform import DataTranformation

from requests import get, post
import json

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_preprocess = preprocessor.transform(features)
            prediction = model.predict(data_preprocess)
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,
        song_name: str,
        artist: str):

        self.song_name = song_name

        self.artist = artist

        self.id = None

        self.explicit = None

        self.popularity = None
    
    def get_song_data(self):
        try:
            # get token to communicate with spotify API
            token = get_token()

            # get body parameters (url and formating token to create header)
            url = f'https://api.spotify.com/v1/search?query={self.song_name} {self.artist}&type=track&limit=1'
            header = get_auth_header(token)

            # get request
            result = get(url, headers = header)
            # loading results
            json_result = json.loads(result.content)

            # storing data that is needed for future
            self.id = json_result['tracks']['items'][0]['id']
            self.popularity = json_result['tracks']['items'][0]['popularity']
            self.explicit = json_result['tracks']['items'][0]['explicit']

            # getting song features
            song_data_url = f'https://api.spotify.com/v1/audio-features/{self.id}'
            data_result = get(song_data_url, headers = header)
            json_result = json.loads(data_result.content)
            # storing spotify data dictionary into data frame
            song_df = pd.DataFrame(json_result, index=[0])
            popularity = [self.popularity]
            explicit = [self.explicit]
            extra_data = {'popularity': popularity, 'explicit':explicit}
            song_df = song_df.assign(**extra_data)
            song_df = song_df[[
                'popularity', 
                'duration_ms', 
                'explicit',
                'danceability', 
                'energy',
                'key', 
                'loudness', 
                'mode',
                'speechiness',
                'acousticness',
                'instrumentalness',
                'liveness',
                'valence',
                'tempo',
                'time_signature'
                ]]
            
            return song_df
        
        except Exception as e:
            raise CustomException(e,sys)