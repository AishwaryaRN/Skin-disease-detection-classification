from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle
from fpdf import FPDF

app = Flask(__name__)

# Load model
model = load_model('skin_model.h5')

# Load class labels
with open('class_labels.pkl', 'rb') as f:
    class_indices = pickle.load(f)

class_names = [None] * len(class_indices)
for key, value in class_indices.items():
    class_names[value] = key

print("Detected classes:", class_names)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['file']
    img_path = os.path.join('static', img.filename)
    img.save(img_path)

    # Preprocess
    img_obj = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img_obj)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    result = model.predict(img_array)
    pred_class_idx = np.argmax(result, axis=1)[0]
    pred_class = class_names[pred_class_idx]
    confidence = round(np.max(result) * 100, 2)

    return render_template('result.html',
                           prediction=pred_class,
                           confidence=confidence,
                           image=img_path)

# PDF generation (optional)
@app.route('/generate_pdf')
def generate_pdf():
    img_path = request.args.get('img')
    pred_class = request.args.get('pred')
    confidence = request.args.get('conf')

    advice_dict = {
        'acne': 'Maintain skin hygiene and avoid oily products.',
        'eczema': 'Keep your skin moisturized and avoid strong soaps.',
        'psoriasis': 'Consult a dermatologist for targeted treatment.',
        'ringworm': 'Keep the affected area clean and dry.',
        'normal': 'Skin appears normal. Maintain good hydration and routine care.'
    }
    advice = advice_dict.get(pred_class, "Consult dermatologist for proper care.")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AI Skin Disease Detection Report", ln=True, align="C")
    pdf.ln(10)
    pdf.image(img_path, x=60, w=90)
    pdf.ln(10)
    pdf.set_font("Arial", '', 14)
    pdf.cell(0, 10, f"Predicted Disease: {pred_class.capitalize()}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence}%", ln=True)
    pdf.multi_cell(0, 10, f"Advice: {advice}")

    pdf_path = "static/skin_report.pdf"
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
