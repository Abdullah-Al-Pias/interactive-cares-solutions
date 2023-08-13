## N.B: Clear documentation and visualization of the outputs are given in the jupyter notebook; this is a python script for the given task

## Here the script works after training the lstm model with the given dataset
## If alredy any saved model weights is found then script works as soon as after loading that model
## The training and evaluation of Distilbert are described in the Jupyter Notebook(Google Colab) as I don't have GPU 

## The script takes argument as: 

## Input directory containing PDF resumes
## Output directory for categorized resumes
## Number of PDF resumes to classify

## run: python script.py --input_dir path/input_dir --output_dir path/output_dir --num_pdfs number_of_resumes_to_be_classified



import os
import re
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#load csv dataset
data = pd.read_csv("dataset/Resume/Resume.csv")  

# Text preprocessing 
def preprocess_text(text):

    # Tokenization
    tokens = word_tokenize(text.lower())

    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    
    return lemmatized_tokens


data['preprocessed_text'] = data['Resume_str'].apply(preprocess_text)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
data['encoded_label'] = label_encoder.fit_transform(data['Category'])

# Split dataset into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Handle class imbalance using resampling
majority_class = train_data['encoded_label'].value_counts().idxmax()
minority_classes = train_data['encoded_label'].value_counts().idxmin()

# Upsample minority classes
upsampled_data = []
for label in train_data['encoded_label'].unique():
    if label == majority_class:
        upsampled_data.append(train_data[train_data['encoded_label'] == label])
    else:
        minority_class_data = train_data[train_data['encoded_label'] == label]
        upsampled_data.append(resample(minority_class_data, replace=True, n_samples=len(train_data[train_data['encoded_label'] == majority_class]), random_state=42))

train_data_upsampled = pd.concat(upsampled_data)

# Tokenize and pad the text data
max_words = 1000  
max_seq_length = 200  
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data_upsampled['preprocessed_text'])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data_upsampled['preprocessed_text']), maxlen=max_seq_length)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['preprocessed_text']), maxlen=max_seq_length)

y_train = tf.keras.utils.to_categorical(train_data_upsampled['encoded_label'], num_classes=len(label_encoder.classes_))
y_test = tf.keras.utils.to_categorical(test_data['encoded_label'], num_classes=len(label_encoder.classes_))

if not os.path.exists('resume_classification_lstm_model.h5'):
    # Define LSTM model
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=max_seq_length),
        LSTM(128, dropout=0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    batch_size = 32
    epochs = 10
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

    model.save('resume_classification_lstm_model.h5')

else:
    model = load_model('resume_classification_lstm_model.h5')

# Define argparse to classify PDF resumes and create output directories
parser = argparse.ArgumentParser(description='Classify PDF resumes and create output directories.')
parser.add_argument('--input_dir', type=str, default='pdf_resumes', help='Input directory containing PDF resumes')
parser.add_argument('--output_dir', type=str, default='classified_resumes', help='Output directory for categorized resumes')
parser.add_argument('--num_pdfs', type=int, default=10, help='Number of PDF resumes to classify')
args = parser.parse_args()

# Create output directory
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Classify random PDF resumes and move to respective output directories
pdf_directory = args.input_dir  
pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith('.pdf')]
selected_pdfs = np.random.choice(pdf_files, size=args.num_pdfs, replace=False)

for pdf_file in selected_pdfs:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()
    pdf_document.close()

    preprocessed_text = preprocess_text(text)
    sequence = pad_sequences(tokenizer.texts_to_sequences([preprocessed_text]), maxlen=max_seq_length)
    input_tensor = tf.convert_to_tensor(sequence)
    
    y_pred = model.predict(input_tensor)
    predicted_label = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
    
    output_category_dir = os.path.join(output_dir, predicted_label[0])
    os.makedirs(output_category_dir, exist_ok=True)
    output_path = os.path.join(output_category_dir, pdf_file)
    
    # Move the PDF to the output directory
    os.rename(pdf_path, output_path)

    #create csv file categorized_resumes
    # categorized_resumes = pd.DataFrame()

# Generate CSV file
output_csv = os.path.join(output_dir, 'categorized_resumes.csv')
results = []
for category in os.listdir(output_dir):
    if os.path.isdir(os.path.join(output_dir, category)):
        for resume_file in os.listdir(os.path.join(output_dir, category)):
            if resume_file.endswith(".pdf"):
                results.append({'filename': resume_file, 'category': category})
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)

print("Randomly selected PDF resumes classified and moved to respective directories.")
