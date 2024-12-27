import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from config import Config

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=Config.IMAGE_SIZE)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return preprocess_input(img)
    
    @staticmethod
    def capture_image(camera, save_path):
        cv2.namedWindow("Live Feed")
        start_time = cv2.getTickCount()
        fps = cv2.getTickFrequency()
        
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame.")
                break

            elapsed_time = (cv2.getTickCount() - start_time) / fps
            if elapsed_time >= Config.CAPTURE_WAIT_TIME:
                cv2.imwrite(save_path, frame)
                print(f"Image captured and saved at {save_path}")
                break
                
            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()