import os
from datetime import datetime
from flask import Flask, render_template, jsonify, url_for
import cv2
from config import Config
from models.vgg_face import VGGFaceModel
from utils.image_processing import ImageProcessor
from utils.verification import FaceVerifier
from utils.logger import VerificationLogger  # Add this import

app = Flask(__name__)

# Configure static folder for images
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'images')
app.config['CAPTURED_FOLDER'] = os.path.join('static', 'captured')

# Ensure the upload and logs folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CAPTURED_FOLDER'], exist_ok=True)
os.makedirs(Config.LOGS_DIR, exist_ok=True)  # Add this line

# Initialize model
try:
    model = VGGFaceModel.build()
    if not VGGFaceModel.load_weights(model, Config.MODEL_WEIGHTS_PATH):
        raise RuntimeError("Failed to load model weights")

    vgg_face_descriptor = VGGFaceModel.get_face_descriptor(model)
    face_verifier = FaceVerifier(vgg_face_descriptor)
except Exception as e:
    print(f"Error initializing model: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture_and_verify():
    try:
        # Create a unique filename for the captured image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        captured_image_filename = f"captured_{timestamp}.jpg"
        captured_image_path = os.path.join(app.config['CAPTURED_FOLDER'], captured_image_filename)
        
        # Ensure the captured folder exists
        os.makedirs(os.path.dirname(captured_image_path), exist_ok=True)
        
        # Initialize camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise Exception("Could not access the camera")
            
        # Capture image
        ImageProcessor.capture_image(camera, captured_image_path)
        camera.release()
        
        if not os.path.exists(captured_image_path):
            raise Exception("Failed to save captured image")

        similar_matches = []
        
        # Iterate through images in the images folder
        images_folder = app.config['UPLOAD_FOLDER']
        for image_file in os.listdir(images_folder):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(images_folder, image_file)
                
                try:
                    result = face_verifier.verify_face(captured_image_path, image_path)
                    
                    if result['is_match']:
                        # Log the verification result
                        VerificationLogger.log_verification(
                            captured_image_path,
                            image_path,
                            result
                        )
                        
                        # Create URL for the matched image
                        image_url = url_for('static', 
                                          filename=f'images/{image_file}',
                                          _external=True)
                        
                        similar_matches.append({
                            "matched_image": image_url,
                            "cosine_similarity": float(result['cosine_similarity']),
                            "euclidean_distance": float(result['euclidean_distance'])
                        })
                except Exception as e:
                    print(f"Error processing image {image_file}: {e}")
                    continue

        return jsonify({
            "status": "success",
            "matches": similar_matches,
            "captured_image": url_for('static', 
                                    filename=f'captured/{captured_image_filename}',
                                    _external=True)
        })

    except Exception as e:
        print(f"Error in capture_and_verify: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)