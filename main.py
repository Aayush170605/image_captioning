#Phase 1 including all the imports and text generation
import string
import numpy as np
import os
import ssl

# --- To ignore the certificate verifications in MacOS ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from PIL import Image
from pickle import dump, load
import time
import tensorflow as tf
from tqdm.auto import tqdm

dataset_text = "Flickr8k_text"
dataset_images = "Flicker8k_Dataset"

# Functions
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption = img_caption.replace("-", " ")
            desc = img_caption.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if (len(word) > 1)]
            desc = [word for word in desc if (word.isalpha())]
            img_caption = ' '.join(desc)
            captions[img][i] = img_caption
    return captions

def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = "\n".join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()

# EXECUTE TEXT GENERATION
print("Generating descriptions.txt...")
filename = dataset_text + "/" + "Flickr8k.token.txt"
descriptions = all_img_captions(filename)
clean_descriptions = cleaning_text(descriptions)
save_descriptions(clean_descriptions, "descriptions.txt")
print("Phase 1 Complete: descriptions.txt created.")

#run this part first

#Phase 2 Feature Extraction (this takes time)
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the Xception Model
print("Loading Xception model...")
model_xception = Xception(include_top=False, pooling='avg')

# Extraction function
def extract_features(directory):
    model = model_xception
    features = {}
    print("Starting feature extraction. This will take 10-20 minutes...")
    # We use tqdm here to see the progress bar
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img

        if not img.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        
        try:
            image = Image.open(filename)
            image = image.resize((299, 299)) # Resize as it accepts only this size
            image = np.array(image)

            if image.shape[2] == 4:
                image = image[..., :3]
            image = np.expand_dims(image, axis=0)
            image = image / 127.5 # Normalize
            image = image - 1.0
            
            feature = model.predict(image, verbose=0)
            features[img] = feature
        except Exception as e:
            print(f"Skipping {img}: {e}")
            continue
            
    return features

# run and save
features = extract_features(dataset_images)
dump(features, open("features.p", "wb"))
print("Phase 2 Complete: features.p created successfully.")

# run phase 2

# Phase 3 includes Training and building the tokenizer(this takes time too)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import add, Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.models import Model

def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(dataset_images, photo))]
    return photos_present

def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words) < 1: continue
        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
    return descriptions

def load_features(photos):
    all_features = load(open("features.p", "rb")) 
    features = {k: all_features[k] for k in photos}
    return features

# Loading everything
print("Loading training data...")
filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)

# Tokenizer
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb')) 
# saves the tokenizer
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary Size: {vocab_size}")

def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

max_length_val = max_length(train_descriptions)
print(f"Max Description Length: {max_length_val}")


def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def data_generator(descriptions, features, tokenizer, max_length):
    def generator():
        while True:
            for key, description_list in descriptions.items():
                if key not in features: continue
                feature = features[key][0]
                input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
                for i in range(len(input_image)):
                    yield {'input_1': input_image[i], 'input_2': input_sequence[i]}, output_word[i]

    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).batch(32)

# Define Model
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

# Start Training
model = define_model(vocab_size, max_length_val)
epochs = 10
steps = len(train_descriptions) // 32

if not os.path.exists("models2"):
    os.mkdir("models2")

print("Starting Training Loop...")
for i in range(epochs):
    dataset = data_generator(train_descriptions, train_features, tokenizer, max_length_val)
    model.fit(dataset, epochs=4, steps_per_epoch=steps, verbose=1)
    model.save("models2/model_" + str(i) + ".h5")
    print(f"Saved model_{i}.h5")
