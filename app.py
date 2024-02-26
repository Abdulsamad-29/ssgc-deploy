from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import os
from main import run

app = Flask(__name__)

@app.route("/Predict", methods = ["GET", "POST"])
def predict():
    try:
        image_path = image_path = request.args.get('image_path')#'D:/Sample_Meter_Images/0010410835-18094950-031411-r-20220920-110507.jpg'#request.args.get('image_path')

        # if not image_path or not os.path.exists(image_path):
        #     return jsonify({'error': 'Invalid or missing image path'})

        result = run(image_path)
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': 'File not found'})


if __name__ == "__main__":
    app.run(debug=True,port=5000)