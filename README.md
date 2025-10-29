🧠 AI-Powered Multi-Disease Medical Diagnosis Assistant
🏥 A Community-Driven Framework for Medical Image Analysis Using Streamlit + TensorFlow + Grad-CAM
🚀 Overview

AI-Powered Multi-Disease Medical Diagnosis Assistant is an open-source diagnostic platform that uses deep learning to analyze medical images (X-rays, CTs, MRIs, etc.) and detect diseases with explainable AI.

This system serves as a foundation framework, allowing contributors to:

🧩 Train and integrate new disease models

🌍 Collaboratively expand the diagnosis capabilities

🔬 Promote open, interpretable medical AI research

Built with:

🧠 TensorFlow/Keras — Deep Learning Models

🌐 Streamlit — Interactive Web Interface

🔍 Grad-CAM — Explainable AI Visualization

🌟 Project Vision

“One framework, many minds — one assistant, many diagnoses.”

This project aims to become a community-curated AI medical assistant, where researchers and developers can contribute new disease models to build a comprehensive diagnostic system.

Core Goals:

🧬 Build a central AI platform for disease detection

🧠 Enable transparent model explainability

🤝 Foster collaborative growth through open-source contributions

⚙️ Support plug-and-play integration for new disease models

🧩 Key Features

✅ Multi-Disease Support — Classify diseases across multiple medical domains
✅ Explainable AI (Grad-CAM) — Visual heatmaps highlight key image regions
✅ Dynamic Model Configuration — Easily expand via disease_info.json
✅ Streamlit Dashboard — Simple, clean, and interactive interface
✅ Community Expandable — Contributors can add new diseases & datasets
✅ Cross-Platform — Works on Windows, Linux, macOS, and cloud platforms

🧠 System Architecture
🖼️ User Uploads Image 
     ↓
🧠 AI Model (Disease-Specific)
     ↓
🔥 Grad-CAM Heatmap Generation
     ↓
📊 Visualization on Streamlit Dashboard


Core Components:

multi_disease_diagnosis.py → Main Streamlit App

disease_info.json → Stores Disease Metadata, Model Paths, and Labels

models/ → Pretrained or community-contributed model files

uploads/ → Temporary user uploads

🧪 Example Workflow

1️⃣ Run the Application

streamlit run multi_disease_diagnosis.py


2️⃣ Upload a Medical Image

Supported formats: X-ray, CT, MRI, Ultrasound, etc.

3️⃣ Select Disease Model

Choose from available pretrained or contributed disease models.

4️⃣ View Results

🧩 Predicted Class

🔥 Grad-CAM Heatmap Visualization

📊 Confidence Scores

⚙️ Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run multi_disease_diagnosis.py

🧬 File Structure
📦 Multi-Disease-Diagnosis
 ┣ 📜 multi_disease_diagnosis.py   # Main Streamlit application
 ┣ 📜 disease_info.json            # Disease configurations and model mappings
 ┣ 📜 requirements.txt             # Dependencies
 ┣ 📜 README.md                    # Project documentation
 ┣ 📂 models/                      # AI models (.keras / .h5)
 ┗ 📂 uploads/                     # Uploaded images

📚 Example — disease_info.json
{
  "Pneumonia": {
    "model_path": "models/pneumonia_model.keras",
    "labels": ["Normal", "Pneumonia"]
  },
  "Brain Tumor": {
    "model_path": "models/brain_tumor_model.keras",
    "labels": ["No Tumor", "Tumor Detected"]
  }
}


🧠 You can easily extend this file by adding new diseases and their trained model paths.

🔬 Grad-CAM Visualization

Grad-CAM helps interpret the model’s decision by showing which regions of the medical image influenced the diagnosis.

Example Output:

🩻 Original X-ray Image

🔥 Grad-CAM Heatmap Overlay

🧩 Disease Prediction: Pneumonia (Confidence: 96.7%)

☁️ Deployment Options

🟢 Streamlit Cloud — One-click deployment
🟣 Hugging Face Spaces — Streamlit + TensorFlow hosting
⚙️ Docker / Local Server — For clinical or institutional setups

🤝 Contribution Guidelines

We welcome researchers, developers, and students to contribute!

How to Contribute a New Disease Model:

🧠 Train your model using TensorFlow/Keras on a medical dataset

💾 Save it as .keras or .h5 under the models/ folder

🧾 Add your disease entry in disease_info.json

🧪 Test locally with Streamlit

🔁 Submit a Pull Request with details of your model

Commit Example:

Add Skin Cancer model and configuration

💡 Future Enhancements

🔹 Support for 3D medical data (CT/MRI Volumes)
🔹 Integration with FHIR/EHR systems for hospital use
🔹 ONNX / TensorFlow Serving for cloud model hosting
🔹 Confidence calibration & ensemble predictions
🔹 Community leaderboard for best-performing models

📄 License

Licensed under the MIT License — free to use, modify, and distribute with attribution.

👨‍💻 Developed By
MedImage-WebAI



🔖 Repository Short Description

🧠 Open-source Streamlit-based framework for explainable AI medical image diagnosis — contributors add new disease models to expand the assistant’s capabilities.
