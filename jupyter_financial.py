import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib as mplot
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mplot.use('svg')
print(plt.get_backend())


# Load dataset
df = pd.read_csv('FinancialPhraseBank.csv', encoding='latin1', header=None)
df.columns = ['sentiment', 'statement']

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['statement'] = df['statement'].apply(clean_text)

# Encode sentiment labels
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])

# Tokenize and pad sequences
tokenizer = Tokenizer(oov_token='<unk>', num_words=3000)
tokenizer.fit_on_texts(df['statement'])
sequences = tokenizer.texts_to_sequences(df['statement'])
max_length = 50
data_x = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
data_y = to_categorical(df['sentiment'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42, stratify=data_y)

# Define model creation function
def create_model(embed_size, conv_filters, lstm_units, dropout_rate):
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embed_size, input_length=max_length),
        Conv1D(filters=conv_filters, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Flatten(),
        Dense(units=3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter configurations
configs = [
    {'embed_size': 50, 'conv_filters': 32, 'lstm_units': 64, 'dropout_rate': 0.3},
    {'embed_size': 100, 'conv_filters': 64, 'lstm_units': 128, 'dropout_rate': 0.4},
    {'embed_size': 200, 'conv_filters': 128, 'lstm_units': 256, 'dropout_rate': 0.5}
]

# Train and evaluate models
results = []
for i, config in enumerate(configs):
    print(f"\nTraining Model {i+1} with config: {config}")
    model = create_model(**config)
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), verbose=2)
    
    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    results.append({'config': config, 'loss': loss, 'accuracy': acc, 'history': history.history})

# Compare results
for i, result in enumerate(results):
    print(f"\nModel {i+1}:")
    print(f"Config: {result['config']}")
    print(f"Loss: {result['loss']:.4f}, Accuracy: {result['accuracy']:.4f}")


plot_dir = "./model_plots"
os.makedirs(plot_dir, exist_ok=True)

for i, result in enumerate(results):
    history = result['history']
    
    # Accuracy plot
    plt.figure()
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Model {i+1} Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_plot_path = os.path.join(plot_dir, f"model_{i+1}_accuracy.svg")
    plt.savefig(accuracy_plot_path)
    plt.close()
    
    # Loss plot
    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f"Model {i+1} Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(plot_dir, f"model_{i+1}_loss.svg")
    plt.savefig(loss_plot_path)
    plt.close()

# List saved plots to confirm
os.listdir(plot_dir)