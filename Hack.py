import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import os

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
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio_file.write(audio_file.read())
    temp_audio_file.close()

    mfcc_features = extract_mfcc_for_prediction(temp_audio_file.name)
    X_pred = np.expand_dims(mfcc_features, axis=0)
    X_pred = np.expand_dims(X_pred, axis=-1)

    # Make predictions
    predictions = model.predict(X_pred)

    # Convert the predictions to emotion labels
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad']
    predicted_emotion_index = np.argmax(predictions)
    predicted_emotion = class_labels[predicted_emotion_index]

    st.subheader("Predicted Emotion:")
    st.write(predicted_emotion)

    # Display the audio waveform
    audio_data, sampling_rate = librosa.load(temp_audio_file.name, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sampling_rate, color='b')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Remove the temporary audio file
    os.unlink(temp_audio_file.name)
