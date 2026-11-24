"""
Flask Web Application & REST API for Spam Mail Detection
Serves the responsive web frontend and exposes live machine-learning prediction
and evaluation metrics endpoints.
"""

import os
import sys
import json
from flask import Flask, request, jsonify, send_from_directory

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.predict import predict_email

app = Flask(__name__, static_folder=".", static_url_path="")

REPORT_PATH = os.path.join(os.path.dirname(__file__), "models", "evaluation_report.json")


@app.route("/")
def index():
    """Serves the main application UI."""
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_files(path):
    """Serves CSS, JS, and asset files."""
    return send_from_directory(".", path)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Expects JSON: { "text": "...raw email content with headers..." }
    Returns real ML pipeline prediction, calibrated probabilities, and top features.
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON request body."}), 400

    raw_text = data.get("text", "")
    result = predict_email(raw_text)

    response_payload = {
        "success": True,
        "prediction": result["prediction"],
        "spam_probability": result["spam_probability"],
        "ham_probability": result["ham_probability"],
        "confidence": result["confidence"],
        "subject": result["subject"],
        "processed_snippet": result["processed_text_snippet"],
        "top_features": result["top_features"],
        "explanation": result.get("explanation", {}),
        "model_name": "Linear SVM (Calibrated)",
        "feature_representation": "TF-IDF Word (1-3) + Char (3-5) N-Grams (40,000 features)",
        "dataset_trained": "Enron Email Spam/Ham Benchmark (29,012 cleaned emails)"
    }
    return jsonify(response_payload)



@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """
    GET /api/metrics
    Returns the real, empirical evaluation metrics for all 4 compared models.
    """
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        return jsonify({"success": True, "report": report})
    return jsonify({"success": False, "error": "Evaluation report not found. Run train_pipeline.py first."}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"\n=======================================================")
    print(f"  Spam Mail Detection ML Server running on http://localhost:{port}")
    print(f"  UI: http://localhost:{port}/")
    print(f"  API: http://localhost:{port}/api/predict")
    print(f"  Metrics: http://localhost:{port}/api/metrics")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)

