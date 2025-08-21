# 🧠 RPE Classification: Age Prediction of RPE Cells

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red?logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning and machine learning project for **predicting the age of Retinal Pigment Epithelium (RPE) cells** using **Convolutional Neural Networks (CNNs)**, **Support Vector Machines (SVMs)**, and **K-Nearest Neighbors (KNN)**.  
This project leverages AI to aid **medical research** and improve understanding of retinal aging patterns.

---

## ✨ Features
- 🧬 **AI-Powered RPE Classification** – Predicts the age group of RPE cells from retinal images.
- 🧠 **Multiple ML Models** – CNN, SVM, and KNN implemented for comparative performance.
- 📈 **High Accuracy** – Achieved **90% accuracy** with CNN.
- 🔍 **Confusion Matrix** for detailed performance analysis.
- 🐍 **Python-Based** and fully reproducible with minimal setup.

---

## 🛠 Tech Stack
| **Category**   | **Technology** |
|---------------|---------------|
| **Language**  | Python |
| **Deep Learning** | PyTorch |
| **Machine Learning** | scikit-learn |
| **Visualization** | Matplotlib |
| **Environment** | Jupyter Notebook / Python Scripts |

---

## 📂 Dataset
- The dataset contains RPE cell images collected from **Emory Eye Center** provided by Dr.Yi Jiang
- The dataset is structured into age groups for classification.

---

## ⚙️ Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/H1BOI/RPE-classification.git
cd RPE-classification
```

### 2️⃣ Set Up Virtual Environment
```bash
python -m venv venv
# On Mac/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage
Run the models individually:

```bash
# Run CNN model
python cnn.py

# Run SVM model
python SVM.py

# Run KNN model
python KNN.py
```

> Make sure your dataset paths (if required by the scripts) are correctly set inside each file.

---

## Results

| **Model** | **Accuracy** |
|-----------|--------------|
| **CNN**   | **90%** ✅ |
| **SVM**   | ~78% |
| **KNN**   | ~72% |

---

## Acknowledgments
Special thanks to:  
- **Dr. Yi Jiang** for guidance and expertise.  
- **Emory Eye Center** for providing RPE cell images for research.

---

## License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more information.
