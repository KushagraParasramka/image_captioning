import streamlit as st
import joblib as jbl
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from textwrap import wrap
import seaborn as sns
import pandas as pd
plt.rcParams['font.size'] = 12
sns.set_style("dark")

# Load the model and tokenizer
caption_model = jbl.load("image_caption_model.joblib")
tokenizer = Tokenizer()

data = pd.read_csv("./archive/captions.txt")
def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]",""))
    data['caption'] = data['caption'].apply(lambda x: x.replace("\s+"," "))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word)>1]))
    data['caption'] = "startseq "+data['caption']+" endseq"
    return data

data = text_preprocessing(data)
captions = data['caption'].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

model = DenseNet201()
fe = Model(inputs=model.input, outputs=model.layers[-2].output)

img_size = 224
features = {}


def idx_to_word(integer,tokenizer):
    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length, features):
    
    feature = features[image]
    in_text = "startseq"
    for i in range(max_length):
        
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature,sequence])
        y_pred = np.argmax(y_pred)
        print(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        
        if word is None:
            break
            
        in_text+= " " + word
        
        if word == 'endseq':
            break
            
    return in_text 

def display_image(image, caption):
    st.image(image, caption=caption, width=300)

def main():
    st.title("Image Captioning App")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = uploaded_file.name
        img = load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption='Uploaded Image', width=300)
        img = img_to_array(img)
        img = img/255.
        img = np.expand_dims(img,axis=0)
        feature = fe.predict(img, verbose=0)
        features[image] = feature
        
        if st.button("Generate Caption"):
            caption = predict_caption(caption_model, image, tokenizer, 34, features)
            caption = caption.lstrip("startseq ").rstrip("endseq")
            st.write("Generated Caption:", caption)

if __name__ == "__main__":
    main()
