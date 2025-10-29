# Medical-Image-Diagnosis-Assistant
# 🩺 AI-Powered Multi-Disease Medical Image Diagnosis System  
### Using Streamlit + TensorFlow + Grad-CAM Explainability  

---

## 🚀 Overview

**Multi-Disease Medical Image Diagnosis** is an **AI-based diagnostic assistant** designed to analyze medical images (X-rays, CT scans, MRIs, etc.) and detect diseases with **deep learning models**.  
The app provides **real-time Grad-CAM heatmaps** to visualize areas of interest — making the predictions **interpretable and clinically useful**.

Built using:
- 🧠 **TensorFlow/Keras** for deep learning  
- 🌐 **Streamlit** for an interactive, browser-based interface  
- 🔍 **Grad-CAM** for model explainability  

---

## 🧩 Key Features

✅ **Multi-Disease Classification** – Supports multiple pretrained disease models (e.g., pneumonia, brain tumor, skin disease, etc.)  
✅ **Explainable AI** – Uses Grad-CAM to show visual attention maps  
✅ **Dynamic JSON Configuration** – Easily manage supported diseases and model paths via `disease_info.json`  
✅ **Streamlit Interface** – Clean, responsive web UI  
✅ **Cross-Platform Deployment** – Runs on Windows, Linux, or cloud platforms (Streamlit Cloud / Hugging Face Spaces)

---

## 🧠 System Architecture

User Uploads Image → Preprocessing → Deep Learning Model Inference → Grad-CAM Heatmap Generation → Visualization on Streamlit Dashboard

markdown
Copy code

**Components:**
- `multi_disease_diagnosis.py` → Main Streamlit app
- `disease_info.json` → Contains model configuration and disease metadata
- `models/` → Folder containing pre-trained model `.h5` files
- `uploads/` → Temporary uploaded images

---

## 🧪 Example Workflow

1. Launch Streamlit:
   ```bash
   streamlit run multi_disease_diagnosis.py
Upload a medical image (X-ray / CT / MRI).

Select the disease model to use.

View:

🧩 Predicted Class

🔥 Grad-CAM Heatmap Visualization

📊 Confidence Scores

⚙️ Installation Guide
1. Clone Repository
bash
Copy code
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
2. Create Virtual Environment (Recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Run Application
bash
Copy code
streamlit run multi_disease_diagnosis.py
🧬 File Structure
graphql
Copy code
📦 Multi-Disease-Diagnosis
 ┣ 📜 multi_disease_diagnosis.py   # Main Streamlit application
 ┣ 📜 disease_info.json            # Disease configuration and model mapping
 ┣ 📜 requirements.txt             # Python dependencies
 ┣ 📜 README.md                    # Documentation
 ┣ 📂 models/                      # Pre-trained TensorFlow models (.h5)
 ┗ 📂 uploads/                     # Uploaded image storage
📚 disease_info.json Example
json
Copy code
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
You can easily extend this by adding new diseases and models to the JSON file.

🔬 Grad-CAM Visualization
Grad-CAM highlights the most influential regions in the image that contributed to the model’s decision.

Example Output:

Original X-ray Image

Grad-CAM Heatmap Overlay

Disease Prediction: Pneumonia (Confidence: 96.7%)

☁️ Deployment Options
🟢 Streamlit Cloud – Easiest option for free hosting

🟣 Hugging Face Spaces – Supports Streamlit + TensorFlow

⚙️ Docker / Local Server – For hospital or internal deployments

🤝 Contribution Guidelines
Fork the repository

Create a new branch (feature/add-new-disease)

Commit your changes with clear messages

Submit a pull request

📄 License
This project is licensed under the MIT License — you are free to use, modify, and distribute it with attribution.

💡 Future Enhancements
🧬 Support for 3D medical data (CT/MRI volumes)

🏥 Integration with FHIR-based patient data

☁️ Real-time model serving via TensorFlow Serving or ONNX

📈 Advanced analytics dashboard

👨‍💻 Developed By
Suraj Poddar
AI & IoT Developer | Open Source Contributor
📧 [your-email@example.com]
🌐 [your-portfolio-or-linkedin-url]
