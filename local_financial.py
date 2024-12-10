import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load dataset
df = pd.read_csv('FinancialPhraseBank.csv', encoding='latin1', header=None)
df.columns = ['sentiment', 'statement']

# Text cleaning function
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
x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.2, random_state=42, stratify=data_y
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['sentiment']),
    y=df['sentiment']
)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class Weights:", class_weights_dict)

# Define model creation function
def create_model(embed_size, conv_filters, lstm_units, dropout_rate):
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embed_size, input_length=max_length),
        Conv1D(filters=conv_filters, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        LSTM(units=lstm_units, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(units=3, activation='softmax', kernel_regularizer=None)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Hyperparameter configurations
configs = [
    {'embed_size': 50, 'conv_filters': 32, 'lstm_units': 64, 'dropout_rate': 0.2},
    {'embed_size': 100, 'conv_filters': 64, 'lstm_units': 128, 'dropout_rate': 0.3},
    {'embed_size': 200, 'conv_filters': 128, 'lstm_units': 256, 'dropout_rate': 0.4}
]

# Train and evaluate models
results = []
for i, config in enumerate(configs):
    print(f"\nTraining Model {i+1} with config: {config}")
    model = create_model(**config)
    history = model.fit(
        x_train, y_train,
        epochs=10, batch_size=128,
        validation_data=(x_test, y_test),
        class_weight=class_weights_dict,
        callbacks=[early_stopping, lr_scheduler],
        verbose=2
    )
    
    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    results.append({'config': config, 'loss': loss, 'accuracy': acc, 'history': history.history})

# Compare results
for i, result in enumerate(results):
    print(f"\nModel {i+1}:")
    print(f"Config: {result['config']}")
    print(f"Loss: {result['loss']:.4f}, Accuracy: {result['accuracy']:.4f}")

# Plot accuracy and loss for all models
for i, result in enumerate(results):
    history = result['history']
    plt.figure()
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Model {i+1} Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f"Model {i+1} Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Final evaluation with classification report and confusion matrix
best_model_index = np.argmax([result['accuracy'] for result in results])
print(f"\nBest Model Config: {results[best_model_index]['config']}")
model = create_model(**results[best_model_index]['config'])
model.fit(
    x_train, y_train,
    epochs=10, batch_size=128,
    validation_data=(x_test, y_test),
    class_weight=class_weights_dict,
    callbacks=[early_stopping, lr_scheduler],
    verbose=2
)

# Predictions and classification report
y_pred = model.predict(x_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.show()
