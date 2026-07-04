# Deepfake Human Detection System

A secure, cloud-independent computer vision application engineered to detect synthetic and GAN-generated facial images entirely at the edge. By combining a lightweight deep learning architecture with a real-time user interface, this system provides a decentralized solution for identity verification and digital forgery detection.

---

## 🚀 Key Features

- **Decoupled Architecture:** Built within a unified Python ecosystem, separating user interface routing from deep learning computation.
- **Immediate Face Gating:** Integrates an upfront OpenCV validation layer to instantly parse images and reject files without human faces, protecting downstream model resources.
- **Edge-Optimized Core:** Leverages a custom-tuned MobileNetV2 backbone via transfer learning to identify micro-pixel anomalies without requiring heavy cloud GPU infrastructure.
- **Strict-Mode Verification:** Configured with an elevated safety threshold (0.75) at the final Sigmoid layer to aggressively minimize critical False Positives in high-security environments.

---

## 🛠️ Tech Stack & Ecosystem

- **Frontend:** Streamlit (Python)
- **Core ML Engine:** TensorFlow / Keras
- **Computer Vision & Processing:** OpenCV, NumPy, Pillow (PIL)
- **Development Environment:** VS Code / Google Colab (Training Phase)

---

## 📦 System Architecture Workflow

1. **Ingestion Layer:** The user uploads a portrait image via the reactive **Streamlit** frontend interface.
2. **Security Gate Module:** **OpenCV** runs a lightweight intensity check using Haar Cascades to confirm the presence of a human face bounding box. 
3. **Preprocessing Module:** **NumPy** and **PIL** resize the validated facial matrix to a standard $224 \times 224 \times 3$ RGB array and normalize pixel integers down to a stable `0.0 - 1.0` floating-point range.
4. **Inference Engine:** **TensorFlow/Keras** evaluates the processed matrix against the pre-trained weights, runs a forward mathematical pass through a Global Average Pooling (GAP) and Dropout head, and outputs a prediction score.

---

## Installation & Local Setup

Follow these steps to set up and run the application locally on your machine using standard consumer hardware:

### 1. Clone the Repository
```bash
git clone [https://github.com/Malavika-sarun/Deepfake-Identity-Detection
](https://github.com/Malavika-sarun/Deepfake-Identity-Detection
.git)
cd Deepfake-Identity-Detection
