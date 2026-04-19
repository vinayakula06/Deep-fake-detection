# DeepShield - AI Deepfake Detection System

A professional deepfake detection system using ensemble EfficientNet-B7 Noisy Student models trained on the DFDC dataset.

## 🏗️ Project Structure

```
dfdc/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── templates/                  # HTML templates
│   ├── index.html             # Main application page
│   └── index_backup.html      # Backup of previous version
│
├── static/                     # Static assets
│   ├── css/
│   │   └── main.css           # Main stylesheet
│   ├── js/
│   │   └── app.js             # Application JavaScript
│   └── assets/                # Images, icons, etc.
│
├── weights/                    # Model weights
│   ├── final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36
│   ├── final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19
│   ├── final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29
│   ├── final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31
│   ├── final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37
│   ├── final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40
│   └── final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23
│
└── uploads/                    # Temporary upload directory
```

## 🚀 Features

### Core Detection Methods
1. **Filename Analysis** - Detects AI signatures in filenames (Gemini, Grok, DALL-E, etc.)
2. **OCR Watermark Detection** - Scans corners and edges for AI generator logos
3. **Neural Network Analysis** - 7-model ensemble using EfficientNet-B7 NS

### Technical Highlights
- **Multi-layer Detection**: Filename → Watermark → Neural Network
- **Batch Face Detection**: MTCNN processes multiple frames simultaneously
- **Optimized Frame Sampling**: 15 frames for videos, adaptive for images
- **GPU Acceleration**: CUDA FP16 autocast with CPU fallback
- **Real-time Progress**: Animated UI with step-by-step feedback

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Tesseract OCR (optional, for watermark detection)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd dfdc
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR** (optional)
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

4. **Place model weights**
Ensure all 7 model weight files are in the `weights/` directory.

## 🎯 Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:8080`

### API Endpoints

#### Health Check
```bash
GET /health
```
Returns server status and loaded models count.

#### Predict
```bash
POST /predict
Content-Type: multipart/form-data
Body: file=<video_or_image>
```

Response:
```json
{
  "verdict": "FAKE",
  "probability": 0.9234,
  "confidence": 0.8468,
  "frame_scores": [0.92, 0.94, ...],
  "frames_analyzed": 15,
  "confident_fakes": 12,
  "risk_level": "HIGH",
  "processing_time": 3.45,
  "detection_method": "NEURAL_NETWORK"
}
```

## 🎨 Frontend Architecture

### CSS Organization (`static/css/main.css`)
- **CSS Variables**: Centralized color scheme and typography
- **Base Styles**: Reset and global styles
- **Components**: Modular, reusable component styles
- **Animations**: Smooth transitions and keyframe animations
- **Responsive**: Mobile-first design with breakpoints

### JavaScript Modules (`static/js/app.js`)
- **State Management**: Application state handling
- **File Handling**: Drag-and-drop and file input
- **API Communication**: Async fetch with error handling
- **UI Updates**: Dynamic DOM manipulation
- **Utilities**: Helper functions for formatting and validation

## 🔧 Configuration

### Model Parameters
```python
INPUT_SIZE = 380        # Input image size
NUM_FRAMES = 15         # Frames to sample from video
BATCH_SIZE = 16         # Inference batch size
```

### Detection Thresholds
```python
CONFIDENT_THRESHOLD = 0.8    # High-confidence fake threshold
FAKE_THRESHOLD = 0.5         # Binary classification threshold
```

## 📊 Performance

- **Inference Time**: ~3-5 seconds per video (GPU)
- **Accuracy**: 0.993 AUC on DFDC validation set
- **Supported Formats**: MP4, MOV, AVI, WEBM, JPG, PNG
- **Max File Size**: 500 MB

## 🛡️ Security Features

- **AI Signature Detection**: Catches AI-generated content by filename
- **Watermark Scanning**: OCR-based logo detection in corners/edges
- **Multi-region Analysis**: Scans 7 regions per frame for watermarks
- **Ensemble Voting**: 7 models reduce false positives

## 🎓 Academic Use

This project is designed as a final year project with:
- Clean, modular architecture
- Comprehensive documentation
- Professional code organization
- Reusable components
- Industry-standard practices

## 🙏 Acknowledgments

- DFDC Dataset
- EfficientNet-B7 Noisy Student
- MTCNN Face Detection
- Flask Framework
