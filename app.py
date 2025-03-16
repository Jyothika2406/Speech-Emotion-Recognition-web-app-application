from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
import uuid
from pydub import AudioSegment

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model and Label Encoder
model_path = r"C:\Users\DELL\OneDrive\Desktop\speech_emotion_model.keras"
label_encoder_path = r"C:\Users\DELL\OneDrive\Desktop\speech emotion music recomandation\label_encoder.pkl"

model = tf.keras.models.load_model(model_path)
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# Feature Extraction
def extract_features(file_path, max_pad_length=100):
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize

        if mfcc.shape[1] < max_pad_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_length]

        return np.expand_dims(mfcc, axis=0)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Prediction Function
def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Error: Could not extract features"

    features = np.expand_dims(features, axis=-1)  # Ensure shape is (1, 40, 100, 1)

    predictions = model.predict(features)
    predicted_class = np.argmax(predictions)
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"})

    if audio_file:
        file_path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + '.wav')
        audio_file.save(file_path)
        
        # Convert to WAV if needed
        if not file_path.endswith('.wav'):
            audio = AudioSegment.from_file(file_path)
            file_path = file_path + ".wav"
            audio.export(file_path, format="wav")

        emotion = predict_emotion(file_path)
        return jsonify({"emotion": emotion})

if __name__ == '__main__':
    app.run(debug=True)
