import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from textblob import TextBlob
import nltk
import requests
from bs4 import BeautifulSoup

nltk.download('punkt')

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

file_name = 'cosmetic.wav' #file name (audio)
Audio(file_name)

data = wavfile.read(file_name)
framerate = data[0]
sounddata = data[1]
time = np.arange(0,len(sounddata))/framerate
input_audio, _ = librosa.load(file_name, sr=16000)
input_values = tokenizer(input_audio, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]
print('Speech(call recording) to text. result :')
print(transcription)


lowercase_txt = transcription.lower()

#cosmetic and clothing
product_words = [
    'shirts', 'pants', 'dresses', 'jackets', 'cotton', 'denim', 'silk', 'wool',
    'hats', 'scarves', 'gloves', 'belts', 'shoes', 'sneakers', 'boots', 'sandals',
    'Nike', 'Gucci', 'Levi\'s', 'small', 'medium', 'large', 'vintage', 'bohemian',
    'formal', 'casual', 'red', 'stripes', 'floral', 'polka dots','lotion',
    'lipstick', 'foundation', 'mascara', 'eyeshadow', 'moisturizer', 'cleanser',
    'serum', 'sunscreen', 'brushes', 'sponges', 'tweezers', 'curlers', 'Maybelline',
    'L\'Oréal', 'MAC', 'perfumes', 'colognes', 'body mists', 'shampoo', 'conditioner',
    'hair oil', 'styling gel', 'aloe vera', 'hyaluronic acid', 'vitamin C','cosmetic','lipstick','lipsticks'
]

def extract_product_words(text):
    product_matches = []
    blob = TextBlob(text)
    for word in blob.noun_phrases:
        for j in word.split():
            if j in product_words:
                product_matches.append(word)
    return product_matches


def scrape_amazon_products(category):
    base_url = 'https://www.flipkart.com'
    search_query = '+'.join(category.split())
    url = f'{base_url}/search?q={search_query}'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        product_links = soup.find_all('a', {'class': 's1Q9rs'})
        product_urls = [base_url + link.get('href') for link in product_links]
        for product_url in product_urls:
            print(product_url)
    else:
        print('Failed to fetch data from Flipkart.')



product_matches = extract_product_words(lowercase_txt)
print('\n\n Noun phrases :')
print(product_matches)


main_list=[]
for i in product_matches:
    main_list.append(i.replace(" ", ""))

for category in main_list:
    scrape_amazon_products(category)
