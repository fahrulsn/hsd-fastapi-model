import tensorflow as tf
import uvicorn
import numpy as np
import pandas as pd
import re
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


app = FastAPI()


class Item(BaseModel):
  teks: str


model = tf.keras.models.load_model('./model/bestmodel.hdf5')
tokenizer = joblib.load('tokenizer.joblib')
alayDict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
alayDict = alayDict.rename(columns={0: 'original', 1: 'replacement'})
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def lowercase(text):
  return text.lower()


def remove_unnecessary_char(text):
  text = re.sub('\n', ' ', text)  # Remove every '\n'
  text = re.sub('\r', ' ', text)  # Remove every '\r'
  text = re.sub(
      '(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
      '', text)  # Remove every URL
  text = re.sub('(?i)rt', ' ', text)  # Remove every retweet symbol
  text = re.sub('@[^\s]+[ \t]', '', text)  # Remove every username
  text = re.sub('(?i)user', '', text)  # Remove every username
  text = re.sub('(?i)url', ' ', text)  # Remove every url
  text = re.sub(r'\\x..', ' ', text)  # Remove every emoji
  text = re.sub('  +', ' ', text)  # Remove extra spaces
  text = re.sub(r'(\w)\1{2,}', r'\1\1',
                text)  #Remove characters repeating more than twice
  return text


def remove_nonaplhanumeric(text):
  text = re.sub('[^0-9a-zA-Z]+', ' ', text)
  return text


alayDictMap = dict(
    zip(alayDict['original'], alayDict['replacement'], strict=False))


def normalize_alay(text):
  return ' '.join([
      alayDictMap[word] if word in alayDictMap else word
      for word in text.split(' ')
  ])

def stemming(text):
  return stemmer.stem(text)

def preprocess(text):
  text = remove_unnecessary_char(text)  # 1
  text = lowercase(text)  # 2
  text = remove_nonaplhanumeric(text)  # 3
  text = normalize_alay(text)  # 4
  text = stemming(text)  # 5
  text = text.strip()
  print(text)
  return text


def processed(text):
  text = preprocess(text)
  text_sekuens = tokenizer.texts_to_sequences([text])
  text_padded = pad_sequences(text_sekuens,
                              maxlen=900,
                              padding='post',
                              truncating='post')
  return text_padded


@app.get("/")
def hello_world():
  return {"message": "Hello, world!"}


@app.post("/predict")
async def classify_text(item: Item):
  teks = item.teks
  print(teks)
  processed_text = processed(teks)
  predictions = model.predict([processed_text])
  hasil = np.argmax(predictions[0])
  print(predictions)
  print(hasil)
  return {"predictions": int(hasil)}


