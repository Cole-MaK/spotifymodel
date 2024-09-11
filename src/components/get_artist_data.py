import os
import json
import base64
import sys
from requests import get, post
import pandas as pd
from dotenv import load_dotenv
from src.exception import CustomException

load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

def get_token():
    auth_string = client_id + ':' + client_secret
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes),'utf-8')

    url = 'https://accounts.spotify.com/api/token' #post request url

    headers = { # post request headers
        "Authorization": "Basic "+ auth_base64,
        "Content-Type": "application/x-www-form-urlencoded" 
    } 

    data = {'grant_type': 'client_credentials'} #post request body

    result = post(url, headers=headers, data = data)
    json_result = json.loads(result.content)
    token = json_result['access_token']
    return token

def get_auth_header(token):
    return {'Authorization': 'Bearer ' + token}

def get_artist_pop(token, query):
    try:
        url = f'https://api.spotify.com/v1/search?query={query}&type=artist&limit=1'
        header = get_auth_header(token)
        result = get(url, headers = header)
        json_result = json.loads(result.content)
        popularity = json_result['artist']['items'][0]['popularity']
        return(popularity)
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    df = pd.read_csv('notebook/spotify.csv')
    artist_columns = df['artists'].unique()
    length = len(artist_columns)
    artist_pop_dict = {}
    token = get_token()
    for i, artist in enumerate(artist_columns):
        artist_1 = artist.split(';')[0]
        popularity = get_artist_pop(token, artist_1)
        artist_pop_dict[artist] = popularity
        print(f'{i}/{length} | {artist}/{artist_1}:{popularity} length of dict:{len(artist_pop_dict)}')
    df = pd.DataFrame(artist_pop_dict)
    df.to_csv('artist_pop.csv', index=False)