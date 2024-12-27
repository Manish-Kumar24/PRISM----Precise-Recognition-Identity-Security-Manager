import numpy as np
from config import Config

class FaceVerifier:
    def __init__(self, vgg_face_descriptor):
        self.vgg_face_descriptor = vgg_face_descriptor
    
    def _cosine_distance(self, source_repr, test_repr):
        a = np.matmul(np.transpose(source_repr), test_repr)
        b = np.sum(np.multiply(source_repr, source_repr))
        c = np.sum(np.multiply(test_repr, test_repr))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
    def _euclidean_distance(self, source_repr, test_repr):
        euclidean_distance = source_repr - test_repr
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        return np.sqrt(euclidean_distance)
    
    def verify_face(self, img1_path, img2_path):
        from utils.image_processing import ImageProcessor
        
        img1_repr = self.vgg_face_descriptor.predict(ImageProcessor.preprocess_image(img1_path))[0, :]
        img2_repr = self.vgg_face_descriptor.predict(ImageProcessor.preprocess_image(img2_path))[0, :]
        
        cosine_similarity = self._cosine_distance(img1_repr, img2_repr)
        euclidean_distance = self._euclidean_distance(img1_repr, img2_repr)
        
        is_same_person = cosine_similarity < Config.SIMILARITY_THRESHOLD
        verification_status = "Verified: They are the same person." if is_same_person else "Unverified: They are not the same person."
        
        return {
            'cosine_similarity': cosine_similarity,
            'euclidean_distance': euclidean_distance,
            'verification_status': verification_status,
            'is_match': is_same_person
        }