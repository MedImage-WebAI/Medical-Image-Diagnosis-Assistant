AI-powered Streamlit app for multi-disease medical image diagnosis with Grad-CAM explainability — community-driven project where contributors can add and train new disease models.

🩺 AI-Powered Multi-Disease Medical Image Diagnosis System
Using Streamlit + TensorFlow + Grad-CAM Explainability
🚀 Overview

Multi-Disease Medical Image Diagnosis is an AI-based diagnostic assistant designed to analyze medical images (X-rays, CT scans, MRIs, etc.) and detect diseases using deep learning models.
The app provides real-time Grad-CAM heatmaps to visualize areas of interest — making the predictions interpretable, explainable, and clinically valuable.

Built using:

🧠 TensorFlow/Keras for deep learning

🌐 Streamlit for an interactive, browser-based interface

🔍 Grad-CAM for model explainability

🧩 Key Features

✅ Multi-Disease Classification – Supports multiple pretrained disease models (e.g., pneumonia, brain tumor, skin disease, etc.)
✅ Explainable AI – Uses Grad-CAM to show visual attention maps
✅ Dynamic JSON Configuration – Easily manage supported diseases and model paths via disease_info.json
✅ Streamlit Interface – Clean, responsive web UI
✅ Cross-Platform Deployment – Works on Windows, Linux, or cloud platforms (Streamlit Cloud / Hugging Face Spaces)

🧠 System Architecture

Flow:
User Uploads Image → Preprocessing → Model Inference → Grad-CAM Generation → Streamlit Visualization

Components:

multi_disease_diagnosis.py → Main Streamlit app

disease_info.json → Contains disease configuration and model metadata

models/ → Folder containing pre-trained model .h5 files

uploads/ → Temporary uploaded images

🧪 Example Workflow

Launch Streamlit:

streamlit run multi_disease_diagnosis.py


Upload a medical image (X-ray / CT / MRI).

Select the disease model to use.

View:

🧩 Predicted Class

🔥 Grad-CAM Heatmap Visualization

📊 Confidence Scores

⚙️ Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Application
streamlit run multi_disease_diagnosis.py

🧬 File Structure
📦 Multi-Disease-Diagnosis
 ┣ 📜 multi_disease_diagnosis.py   # Main Streamlit application
 ┣ 📜 disease_info.json            # Disease configuration and model mapping
 ┣ 📜 requirements.txt             # Python dependencies
 ┣ 📜 README.md                    # Documentation
 ┣ 📂 models/                      # Pre-trained TensorFlow models (.h5)
 ┗ 📂 uploads/                     # Uploaded image storage

📚 Example disease_info.json
{
  "pneumonia": {
    "model_path": "models/pneumonia_model.h5",
    "labels": ["Normal", "Pneumonia"]
  },
  "brain_tumor": {
    "model_path": "models/brain_tumor_model.h5",
    "labels": ["No Tumor", "Tumor Detected"]
  }
}


👉 To add a new disease, simply extend this file with a new entry — no code changes needed!

🔬 Grad-CAM Visualization

Grad-CAM highlights the regions that most influenced the model’s decision — making AI predictions transparent and explainable.

Example Output:

Original X-ray Image

Grad-CAM Heatmap Overlay

Prediction: Pneumonia (Confidence: 96.7%)

☁️ Deployment Options

🟢 Streamlit Cloud – Easiest for free hosting
🟣 Hugging Face Spaces – Streamlit + TensorFlow ready
⚙️ Docker / Local Server – For hospital or internal deployment

🤝 Contribution Guidelines

This project is designed to be community-driven and expandable.
The goal is to create a large-scale AI diagnostic platform that supports multiple diseases — contributed and trained by the open-source community.

🧩 How You Can Contribute

Add a new disease:

Train a new model (.h5 file) using TensorFlow/Keras for your target disease.

Add an entry for it in disease_info.json (with model path and labels).

Place the model in the models/ folder.

Improve existing models or UI:

Enhance Grad-CAM visualization

Improve preprocessing pipelines

Add new explainability tools or visualization options

Submit your contribution:

Fork this repository

Create a new branch (feature/add-new-disease-<name>)

Commit your changes

Submit a pull request

Every merged contribution helps this project evolve into a comprehensive AI-powered Medical Diagnosis Assistant 🧠

📄 License

Licensed under the MIT License — feel free to use, modify, and distribute with proper attribution.

💡 Future Enhancements

🧬 3D CT/MRI volume support

🏥 Integration with hospital data systems (FHIR/HL7)

☁️ TensorFlow Serving / ONNX for real-time inference

📈 Dashboard for analytics and disease model benchmarking

👨‍💻 Developed By

Suraj Poddar
AI & IoT Developer | Open Source Contributor
📧 your-email@example.com

🌐 [Your LinkedIn or Portfolio Link]
