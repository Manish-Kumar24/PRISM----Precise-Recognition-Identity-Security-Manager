# 🌈 PRISM - Precise Recognition & Identity Security Manager

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)

<img src="/api/placeholder/800/400" alt="PRISM Demo" />

PRISM (Precise Recognition & Identity Security Manager) is a sophisticated facial recognition system built with Flask and VGGFace model. This system captures images through a webcam and compares them against a database of known faces, providing similarity metrics and logging verification results.

## 🌟 Features

- 📸 Real-time face capture using webcam
- 🔍 Face detection and recognition using VGGFace model
- 📊 Similarity metrics (Cosine similarity and Euclidean distance)
- 📝 Automated logging of verification results
- 💻 Web-based user interface
- 🗃️ Excel-based logging system

## 🗂️ Project Structure

```
PRISM/
├── logs/
│   └── excel.xlsx          # Verification logs
├── models/
│   ├── vgg_face_weights.h5 # Pre-trained model weights
│   └── vgg_face.py         # VGGFace model implementation
├── static/
│   ├── captured/           # Captured images storage
│   └── images/             # Reference images database
├── templates/
│   └── index.html          # Web interface template
├── utils/
│   ├── image_preprocessing.py  # Image processing utilities
│   ├── logger.py              # Logging functionality
│   └── verification.py        # Face verification logic
├── app.py                  # Flask application
├── config.py              # Configuration settings
└── requirements.txt       # Project dependencies
```

## 🔧 Prerequisites

- Python 3.8+
- Webcam access
- Sufficient storage for image processing
- CUDA-compatible GPU (recommended for better performance)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Manish-Kumar24/PRISM--Precise-Recognition-Identity-Security-Manager.git
cd PRISM--Precise-Recognition-Identity-Security-Manager
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download VGGFace weights:
- Place the `vgg_face_weights.h5` file in the `models/` directory
- Ensure the weights file matches the model architecture in `vgg_face.py`

## 🚀 Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. The system will:
   - Capture your face through the webcam
   - Compare it against the database
   - Display matching results with similarity scores
   - Log the verification results

## 📝 Configuration

Edit `config.py` to modify:
- Model paths
- Threshold values
- Logging settings
- Image processing parameters

## 📊 Logging

The system maintains logs in an Excel file (`logs/face_recognition_verification_log.xlsx`) containing:
- Timestamp of verification
- Paths to compared images
- Similarity metrics
- Verification results

## 🛠️ API Endpoints

### `/` (GET)
- Renders the main interface

### `/capture` (POST)
- Captures and processes face images
- Returns JSON with matching results:
  ```json
  {
    "status": "success",
    "matches": [
      {
        "matched_image": "image_url",
        "cosine_similarity": 0.95,
        "euclidean_distance": 0.05
      }
    ],
    "captured_image": "captured_image_url"
  }
  ```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Notes

- Ensure proper lighting conditions for better accuracy
- Regular updates to the reference image database recommended
- Check logs periodically for system performance
- Backup the logs directory regularly

## 👥 Contact

Manish Kumar - [manishkumar202209@gmail.com]

Project Link: [https://github.com/Manish-Kumar24/PRISM--Precise-Recognition-Identity-Security-Manager.git](https://github.com/Manish-Kumar24/PRISM--Precise-Recognition-Identity-Security-Manager.git)