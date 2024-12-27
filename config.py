import os

class Config:
    # Create a logs directory in your project root
    LOGS_DIR = "logs"
    FILE_NAME = os.path.join(LOGS_DIR, "face_recognition_verification_log.xlsx")
    IMAGES_FOLDER = os.path.join('static', 'images')
    CAPTURED_FOLDER = os.path.join('static', 'captured')
    MODEL_WEIGHTS_PATH = "models/vgg_face_weights.h5"
    SIMILARITY_THRESHOLD = 0.40
    IMAGE_SIZE = (224, 224)
    CAPTURE_WAIT_TIME = 5  # seconds