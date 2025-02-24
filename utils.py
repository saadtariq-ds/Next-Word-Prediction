import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Load the Model
model = load_model('next_word_prediction_lstm.keras')

# Load Tokenizer
with open(file='tokenizer.pickle', mode='rb') as file:
    tokenizer = pickle.load(file=file)


def predict_next_word(model, tokenizer, text, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length-1):]

    token_list = pad_sequences(
        sequences=[token_list],
        maxlen=max_sequence_length-1,
        padding='pre',
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None