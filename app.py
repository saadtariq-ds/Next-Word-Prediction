import streamlit as st
from utils import predict_next_word, model, tokenizer

st.title("Next Word Prediction")
input_text = st.text_input("Enter the Sequence of words")

if st.button("Predict Next Word"):
    max_sequence_length = model.input_shape[1] + 1
    next_word = predict_next_word(
        model=model,
        tokenizer=tokenizer,
        text=input_text,
        max_sequence_length=max_sequence_length
    )

    st.write(f"Next Word: {next_word}")