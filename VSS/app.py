# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import uuid

# app = Flask(__name__)
# CORS(app)

# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # API to upload video
# @app.route('/upload_video', methods=['POST'])
# def upload_video():
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400

#     video = request.files['video']

#     if video.filename == '':
#         return jsonify({'error': 'No selected video file'}), 400

#     if video and allowed_file(video.filename):
#         video_id = str(uuid.uuid4())
#         filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}.mp4")
#         video.save(filename)

#         video_url = f"http://127.0.0.1:5000/video/{video_id}"

#         return jsonify({'message': 'Video uploaded successfully', 'video_url': video_url}), 200

#     return jsonify({'error': 'Invalid file format. Please upload a valid MP4 video file.'}), 400

# # Assuming you have images in the 'images' directory
# @app.route('/images/<filename>')
# def serve_image(filename):
#     return send_from_directory('images', filename)
# # API to get all images
# @app.route('/get_all_images', methods=['GET'])
# def get_all_images():
#     image_folder = 'images'
#     image_names = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
#     return jsonify({'images': image_names})

# # Helper function to check allowed file format
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4'}

# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run(debug=True)


from flask import send_from_directory



from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import os
import uuid
from test import *
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import mobilenet_v2  # Assuming MobileNetV2 was used for training
from prediction.process_video_upload import upload_video_process
from prediction.utils import *
loaded_model = load_model('fire_classification_model.h5')
app = Flask(__name__)
CORS(app)
loaded_model = load_model('fire_classification_model.h5')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def preprocess_image(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize to target size
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = mobilenet_v2.preprocess_input(frame)

    return frame


def gen_frames(model):
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to capture frame from the camera.")
            break
        prediction = start_prediction(frame,model)

        fire_text = "Fire: {:.2f}%".format(prediction * 100)
        cv2.putText(frame, fire_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)

 

        if not ret:
            print("Failed to encode frame to JPEG.")
            break
        frame_bytes = buffer.tobytes()
        print("Frame captured and encoded successfully.")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():

    return Response(gen_frames(loaded_model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')



# API to upload video
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']

    if video.filename == '':
        return jsonify({'error': 'No selected video file'}), 400

    if video and allowed_file(video.filename):
        video_id = str(uuid.uuid4())
        filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}.mp4")
        video.save(filename)

        upload_video_process(filename, loaded_model)

        video_url = f"http://127.0.0.1:5000/video/{video_id}"

        return jsonify({'message': 'Video uploaded successfully', 'video_url': video_url}), 200

    return jsonify({'error': 'Invalid file format. Please upload a valid MP4 video file.'}), 400



# # API to get all images
# @app.route('/get_all_images', methods=['GET'])
# def get_all_images():
    

#     image_folder = 'images'

#     image_names = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
#     return jsonify({'images': image_names})

# # Helper function to check allowed file format
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4'}



# @app.route('/images/<path:filename>')
# def get_image(filename):
#     print(f"Request Hit...{filename}")
#     return send_from_directory('images', filename)
# API to get all images
# @app.route('/get_all_images', methods=['GET'])
# def get_all_images():
#     image_folder = 'images'
#     image_names = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
#     return jsonify({'images': image_names})
from flask import Flask, jsonify, send_from_directory
import os

app = Flask(__name__)

@app.route('/get_all_images', methods=['GET'])
def get_all_images():
    images_folder = 'images'
    if not os.path.exists(images_folder):
        return jsonify({'error': 'Images folder not found'}), 404
    
    image_files = os.listdir(images_folder)
    return jsonify({'images': image_files})

@app.route('/images/<path:image_name>', methods=['GET'])
def get_image(image_name):
    images_folder = 'images'
    return send_from_directory(images_folder, image_name)



if __name__ == '__main__':
   
    
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)




