import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pickle import load
import tensorflow as tf

# Standardized imports for Python 3.13
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image!")
        return None
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 
    image = image - 1.0
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Load the tokenizer
tokenizer = load(open("tokenizer.p", "rb"))
vocab_size = len(tokenizer.word_index) + 1
max_length = 32 # Matches training [cite: 19]

# model weights
model = define_model(vocab_size, max_length)
model.load_weights('models2/model_9.h5') # FIXED PATH 

xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
if photo is not None:
    img = Image.open(img_path)
    description = generate_desc(model, tokenizer, photo, max_length)
    
    # to make the output look professional
    clean_desc = description.replace('start ', '').replace(' end', '')
    print("\n--- GENERATED CAPTION ---")
    print(clean_desc.capitalize())
    print("------------------------------------------------\n")
    
    plt.imshow(img)
    plt.show()

    #To test the model use any image from the Flickr8k_text/Flickr_8k.testImages.txt and paste it into this command 
    #python test.py -i Flicker8k_Dataset/1317292658_ba29330a0b.jpg