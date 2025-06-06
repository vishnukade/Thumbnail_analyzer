import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # Consistent use of tensorflow.keras
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
import pytesseract
import requests
import base64
import json
import os
import re

class EnsembleModel:
    def __init__(self, model1, model2, model3):
        self.models = [
            tf.keras.models.load_model(model1),
            tf.keras.models.load_model(model2),
            tf.keras.models.load_model(model3)
        ]

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def ensemble_predict(self, img_path):
        input_img = self.preprocess_image(img_path)
        predictions = []

        for i, model in enumerate(self.models):
            pred = model.predict(input_img, verbose=0)  # e.g., [[0.8]]
            predictions.append(pred)
            print(f"Model {i+1} Prediction: {round(float(pred[0][0]) * 100, 2)}%")

        avg_prediction = np.mean(predictions, axis=0)  # [[avg_prob]]
        confidence = float(avg_prediction[0][0])
        predicted_class = 1 if confidence > 0.5 else 0

        result = "Good" if predicted_class == 1 else "Bad"
        print(f"\n🔮 Final Prediction: {result} ({round(confidence * 100, 2)}%)")
        return [predicted_class, round(confidence * 100, 2)]
    
class ImageFeatureExtractor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mobilenet = tf.keras.applications.MobileNetV2(weights="imagenet")

    def brightness_score(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])  # V channel
        return round(brightness, 2)

    def clarity_score(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return round(laplacian_var, 2)

    def sharpness_score(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sharpness = np.mean(np.sqrt(sobelx**2 + sobely**2))
        return round(sharpness, 2)

    def text_presence(self, img):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(pil_img)
        return 1 if text.strip() else 0

    def person_presence(self, img):
        resized = cv2.resize(img, (224, 224))
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(resized.astype(np.float32), axis=0))
        preds = self.mobilenet.predict(arr)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]
        for (_, label, _) in decoded:
            if any(word in label.lower() for word in ["person", "man", "woman", "boy", "girl"]):
                return 1
        return 0

    def standardize_features(self, raw_features, max_clarity=5000, max_sharpness=1500):
        brightness = raw_features['brightness_score']
        clarity = min(raw_features['clarity_score'] / max_clarity * 100, 100)
        sharpness = min(raw_features['sharpness_score'] / max_sharpness * 100, 100)
        text = raw_features['text_presence'] * 100
        person = raw_features['person_presence'] * 100

        return {
            'brightness': round(brightness, 2),
            'clarity': round(clarity, 2),
            'sharpness': round(sharpness, 2),
            'has_text': text,
            'has_person': person
        }

    def extract_all(self, image_path):
        img = cv2.imread(image_path)
        raw = {
            "brightness_score": self.brightness_score(img),
            "clarity_score": self.clarity_score(img),
            "sharpness_score": self.sharpness_score(img),
            "text_presence": self.text_presence(img),
            "person_presence": self.person_presence(img)
        }
        standardized = self.standardize_features(raw)
        return {
            "standardized": standardized
        }

class GeminiThumbnailAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        self.prompt = """Analyze this image as a YouTube thumbnail.
Provide a structured response with the following keys:
- score: the click-through potential score (out of 100).
- strengths: a list of strengths.
- weaknesses: a list of weaknesses.
- recommendations: specific recommendations for improvement, with clear steps for better visual appeal.
Ensure the response is in JSON format."""

    def encode_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")

    def analyze(self, image_path):
        encoded_image = self.encode_image(image_path)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self.prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpg",
                                "data": encoded_image
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(self.api_url, json=payload)
        if response.status_code == 200:
            try:
                result = response.json()
                raw_response = result['candidates'][0]['content']['parts'][0]['text']
                cleaned = re.sub(r"^```(?:json)?|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
                data = json.loads(cleaned)
                return {
                    "score": data.get("score", 0),
                    "strengths": data.get("strengths", []),
                    "weaknesses": data.get("weaknesses", []),
                    "recommendations": data.get("recommendations", [])
                }
            except Exception as e:
                print(f"❌ Failed to parse Gemini response: {e}")
                return None
        else:
            print("❌ API Error:", response.status_code, response.text)
            return None

class ThumbnailScoreAggregator:
    def __init__(self, ensemble_weight=0.3, feature_weight=0.2, gemini_weight=0.5):
        self.weights = {
            "ensemble": ensemble_weight,
            "features": feature_weight,
            "gemini": gemini_weight
        }

    def compute_feature_score(self, features):
        score = (
            features["brightness"] * 0.25 +
            features["clarity"] * 0.25 +
            features["sharpness"] * 0.25 +
            features["has_text"] * 10 +
            features["has_person"] * 15
        )
        return min(score, 100)

    def compute_ensemble_score(self, good_bad_score, confidence):
        return confidence if good_bad_score == 1 else 100 - confidence

    def aggregate_score(self, ensemble_result, features, gemini_score):
        good_bad_score, confidence = ensemble_result

        feature_score = self.compute_feature_score(features)
        ensemble_score = self.compute_ensemble_score(good_bad_score, confidence)

        final_score = (
            ensemble_score * self.weights["ensemble"] +
            feature_score * self.weights["features"] +
            gemini_score * self.weights["gemini"]
        )
        return round(final_score, 2)
