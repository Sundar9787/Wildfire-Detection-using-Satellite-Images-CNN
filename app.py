from flask import Flask, render_template, request, jsonify, send_file, session
import numpy as np
import os
import logging
import base64
import cv2
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "temp"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the wildfire classification model
model_path = "model\wildfire_model.h5"
try:
    model = load_model(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Define categories
categories = ["No Wildfire", "Wildfire"]

# Define precautions based on predictions
disaster_info = {
    "No Wildfire": {
        "Description": "No visible signs of wildfire detected.",
        "Precautions": "Stay alert and follow weather advisories."
    },
    "Wildfire": {
        "Description": "Signs of wildfire detected in the image.",
        "Precautions": "Evacuate if necessary, follow emergency protocols, and wear protective masks."
    }
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_file = request.files["file"]
        if img_file.filename == "":
            return jsonify({"error": "No file selected"})
        
        filename = secure_filename(img_file.filename)
        temp_img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        img_file.save(temp_img_path)

        # Preprocess the image
        img = cv2.imread(temp_img_path)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        predicted_class = categories[int(prediction[0][0] > 0.5)]
        confidence = round(float(prediction[0][0]) * 100, 2)

        # Fetch precautions
        description = disaster_info[predicted_class]["Description"]
        precautions = disaster_info[predicted_class]["Precautions"]

        # Encode image to base64
        with open(temp_img_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
        os.remove(temp_img_path)

        return render_template(
            "predicted.html", 
            predicted=predicted_class, 
            confidence=confidence,
            image=encoded_image, 
            description=description, 
            precautions=precautions
        )
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": "Error processing image"}), 500

@app.route("/download_report")
def download_report():
    predicted_class = session.get("predicted_class")
    if predicted_class:
        description = disaster_info[predicted_class]["Description"]
        precautions = disaster_info[predicted_class]["Precautions"]

        report_content = f"Predicted Status: {predicted_class}\n\nDescription:\n{description}\n\nPrecautions:\n{precautions}"
        report_filename = "report.txt"
        
        with open(report_filename, "w") as report_file:
            report_file.write(report_content)
        
        return send_file(report_filename, as_attachment=True)
    else:
        return "Prediction not found."

if __name__ == "__main__":
    app.run(debug=True)