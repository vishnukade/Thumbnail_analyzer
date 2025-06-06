from flask import Flask, request, jsonify, render_template, redirect, url_for
from module import EnsembleModel, ImageFeatureExtractor, ThumbnailScoreAggregator, GeminiThumbnailAnalyzer
import os

app = Flask(__name__, static_folder="static")
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models once
ensemble_model = EnsembleModel("models/cnn_model.keras", "models/vgg.keras", "models/xception.keras")
feature_extractor = ImageFeatureExtractor()
score_aggregator = ThumbnailScoreAggregator()
gemini_model = GeminiThumbnailAnalyzer("AIzaSyCSjddmdVaePqP4XbtwraX0_NO16zi9Oro")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("thumbnail")
        if not file:
            return "No file uploaded", 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Predict using ensemble model and extract features
        ensemble_result = ensemble_model.ensemble_predict(filepath)
        features = feature_extractor.extract_all(filepath)

        # Run Gemini analysis using the actual file path
        gemini_result = gemini_model.analyze(filepath)
        gemini_score = gemini_result["score"] if gemini_result else 0

        # Access standardized features
        standardized_features = features["standardized"]

        # Calculate final score
        final_score = score_aggregator.aggregate_score(ensemble_result, standardized_features, gemini_score)

        return render_template("result.html", score=final_score, gemini=gemini_result)

    return render_template("upload.html")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
