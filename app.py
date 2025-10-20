from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('skin_model.h5')

# Map class indices to actual disease names (in the same order as your folders)
class_names = ['acne', 'eczema', 'psoriasis', 'ringworm', 'normal']  # update based on your dataset

#  Load trained model
model = load_model('skin_model.h5')
import os

train_dir = 'skin dataset/dataset/train'
class_names = sorted(os.listdir(train_dir))  # automatically gets all 7 folder names
print("Detected classes:", class_names)

#  Home page
@app.route('/')
def home():
    return render_template('index.html')

#  Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['file']                 # get uploaded file
    img_path = os.path.join('static', img.filename)
    img.save(img_path)                          # save it in static folder

    # preprocess the image
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255
    img_array = np.expand_dims(img_array, axis=0)

    # make prediction
    result = model.predict(img_array)
    pred_class_idx = np.argmax(result, axis=1)[0]
    pred_class = class_names[pred_class_idx]  # get human-readable class name

    return render_template('result.html', prediction=pred_class, image=img_path)

if __name__ == '__main__':
    app.run(debug=True)
