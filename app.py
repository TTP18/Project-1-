from flask import Flask, render_template, request, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import uuid

# Load model
model = tf.keras.models.load_model("C:/Users/ADMIN/Desktop/Pattern Recognition/cnn_fruits_vegetables.h5")

# Load class indices
class_indices = np.load("C:/Users/ADMIN/Desktop/Pattern Recognition/class_indices.npy", allow_pickle=True).item()
class_labels = list(class_indices.keys())

app = Flask(__name__)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return class_labels[class_index], float(np.max(prediction))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    uploaded_image = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            # ✅ Luôn lấy absolute path để lưu
            static_dir = os.path.join(app.root_path, "static")
            os.makedirs(static_dir, exist_ok=True)

            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(static_dir, filename)

            # Lưu file
            file.save(filepath)

            # Xóa file cũ trong static (trừ file vừa lưu)
            for f in os.listdir(static_dir):
                file_path = os.path.join(static_dir, f)
                if os.path.isfile(file_path) and f != filename:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print("Không thể xóa:", file_path, e)

            # Dự đoán
            label, conf = predict_image(filepath)
            result = label
            confidence = round(conf * 100, 2)

            # Tạo URL cho ảnh
            uploaded_image = url_for("static", filename=filename)
            print("Ảnh hiển thị:", uploaded_image)

    return render_template("index.html", result=result, confidence=confidence, image=uploaded_image)

if __name__ == "__main__":
    app.run(debug=True)
