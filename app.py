
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = load_model("models/brain_tumor_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=(224,224)
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0

    pred = model.predict(img_array)

    confidence = round(float(pred[0][0])*100,2)

    if pred[0][0] > 0.5:
        prediction = "Tumor Detected"
    else:
        prediction = "No Tumor"
        confidence = round((1-float(pred[0][0]))*100,2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)