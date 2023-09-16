import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# Function to extract MFCC features from an audio file
def extract_mfcc_for_prediction(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Load the pre-trained model
model = tf.keras.models.load_model('emotion_model.h5')

# Streamlit UI
st.title("Audio Emotion Detection")

# Upload an audio file
audio_file = st.file_uploader("Upload an audio file (in WAV format)", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')

    # Extract MFCC features and make a prediction
    mfcc_features = extract_mfcc_for_prediction(audio_file)
    X_pred = np.expand_dims(mfcc_features, axis=0)
    X_pred = np.expand_dims(X_pred, axis=-1)

    # Make predictions
    predictions = model.predict(X_pred)

    # Convert the predictions to emotion labels
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_emotion_index = np.argmax(predictions)
    predicted_emotion = class_labels[predicted_emotion_index]

    st.subheader("Predicted Emotion:")
    st.write(predicted_emotion)
