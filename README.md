
# UPA Project: Breast Cancer Wisconsin Diagnostic

## 📋 Overview
This project implements the **Unsupervised Prediction Alignment (UPA)** algorithm based on the research paper published in *Nature Communications (2023)*. The algorithm automatically corrects performance drift caused by acquisition shifts in medical image classification without requiring new labeled data.

## 🎯 Key Features
- **Automatic Correction**: Corrects performance drift without human intervention
- **No Label Requirement**: Works with unlabeled data only
- **Fast Adaptation**: Real-time alignment (<1 second)
- **Preserves Diagnostic Power**: Maintains ROC-AUC while balancing sensitivity/specificity
- **Web Interface**: User-friendly Flask-based web application

## 📊 Results Summary

### Performance Comparison

| Metric | Original | After Shift (0.4) | After UPA | Improvement |
|--------|----------|-------------------|-----------|-------------|
| Sensitivity | 0.872 | 0.761 | 0.805 | +0.044 |
| Specificity | 0.856 | 0.692 | 0.812 | +0.120 |
| Accuracy | 0.864 | 0.727 | 0.809 | +0.082 |
| SEN-SPC Diff | 0.016 | 0.069 | -0.007 | -0.076 |

**Key Findings:**
- UPA recovered **39.6%** of sensitivity loss
- UPA recovered **73.2%** of specificity loss
- Achieved near-balanced sensitivity/specificity (0.805 vs 0.812)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/mahdieslaminet/Medical-AI-Performance-Drift_Correction.git
cd upa_project

# 2. Create virtual environment (recommended)
python -m venv 'my_venv'

# 3. Activate virtual environment
# On Windows:
my_venv\Scripts\activate
# On macOS/Linux:
source my_venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download dataset
# Option 1: Download manually from Kaggle
# Option 2: Use the provided sample data
```

### 📦 Dependencies

The project requires the following Python packages:

```txt
# Core Data Processing
pandas>=2.0.3
numpy>=1.24.3

# Machine Learning
scikit-learn>=1.3.0
scipy>=1.10.1

# Visualization
matplotlib>=3.7.1
seaborn>=0.12.2

# Web Application
flask>=2.3.2

# Optional: Jupyter support
jupyter>=1.0.0
ipykernel>=6.0.0
```

## 📁 Project Structure

```
upa_web_app/
├── templates/
│   ├── index.html         # صفحه اصلی
│   ├── results.html       # صفحه نتایج
│   └── about.html         # صفحه درباره
├── static/
│   ├── css/
│   │   └── style.css      # استایل‌ها
│   └── images/            # عکس‌ها
├── models/
│   └── upa_model.pkl      # مدل ذخیره شده
├── uploads/
├── app.py                 # کد اصلی Flask
├── data.csv               # دیتاست
└── requirements.txt       # نیازمندی‌ها
├── upa_final.py               # Breast Cancer 
├── upa_results.png
└── upa_notebook.ipynb
```

## 🖥️ Web Application


### Running the Web App

```bash
# From the project root directory
python app.py

# The application will be available at:
# http://localhost:5000
```

### Web App Features

1. **Main Dashboard**
   - Real-time predictions on new samples
   - Shift simulation with severity control
   - Performance monitoring

2. **Results Visualization**
   - Interactive charts and graphs
   - Performance comparison tables
   - Downloadable reports (PDF/PNG)

3. **Scenario Testing**
   - New site deployment simulation
   - Gradual transition scenarios
   - Software update impact analysis

## 📓 Jupyter Notebook

### Running the Notebook

```bash
# 1. Install Jupyter if not already installed
pip install jupyter

# 2. Launch Jupyter notebook
jupyter notebook upa_notebook.ipynb

# 3. Run all cells sequentially
```

### Notebook Contents

The notebook (`upa_notebook.ipynb`) contains:

1. **Data Loading & Exploration**
   - Dataset overview
   - Class distribution analysis
   - Feature statistics

2. **Preprocessing Pipeline**
   - Data cleaning
   - Feature standardization
   - Train-test splitting

3. **Model Training**
   - Random Forest classifier
   - Balanced class weights
   - Hyperparameter tuning

4. **Acquisition Shift Simulation**
   - Intensity shift
   - Noise shift
   - Resolution shift
   - Mixed shift scenarios

5. **UPA Implementation**
   - Prediction distribution alignment
   - CDF-based mapping
   - Real-time correction

6. **Evaluation & Visualization**
   - Performance metrics calculation
   - ROC curve analysis
   - Confusion matrices
   - Comparative visualizations


## 📈 Experimental Results

### Dataset
- **Name**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: UCI Machine Learning Repository
- **Samples**: 569 (357 benign, 212 malignant)
- **Features**: 30 numerical features from cell nuclei

### Performance Metrics

**Original Performance:**
- Sensitivity: 87.2%
- Specificity: 85.6%
- Accuracy: 86.4%
- AUC: 0.923

**After Acquisition Shift (severity=0.4):**
- Sensitivity: 76.1% (↓11.1%)
- Specificity: 69.2% (↓16.4%)
- Accuracy: 72.7% (↓13.7%)

**After UPA Correction:**
- Sensitivity: 80.5% (↑4.4%)
- Specificity: 81.2% (↑12.0%)
- Accuracy: 80.9% (↑8.2%)

## 📚 References

### Academic Paper
- **Main Reference**: "Automatic correction of performance drift under acquisition shift in medical image classification"
- **Journal**: Nature Communications (2023)
- **DOI**: [10.1038/s41467-023-42396-y](https://doi.org/10.1038/s41467-023-42396-y)

### Dataset Source
- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: UCI Machine Learning Repository
- **Samples**: 569 cases
- **Classes**: Benign (357), Malignant (212)
- **Features**: 30 numerical features computed from digitized images
- **Source**: UCI Machine Learning Repository
- **Kaggle**: [Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

- **Translated official paper in Persian**: [Translated](https://drive.google.com/file/d/1A1Ch2KI2Ra-onLhDbme2d0Hpod_dbS7z/view?usp=sharing)


## ⚠️ Limitations & Assumptions

### Algorithmic Limitations
1. Assumes ROC-AUC preservation between domains
2. Requires disease prevalence consistency
3. Focuses only on threshold shift (not other dataset shifts)

### Implementation Limitations
1. Uses relatively small dataset
2. Simplified acquisition shift simulation
3. No access to real multi-scanner data

### Requirements
1. Minimum 250-500 unlabeled samples for alignment
2. Reference data quality affects performance
3. Computational resources for real-time processing

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

**Note**: This is a simplified implementation for educational purposes. Clinical applications would require more rigorous validation and testing.


## 👥 Authors

- **Parinaz Torabi** 
- **University**: Science and Research Branch of Islamic Azad University
- **Professor**: Mahdi Eslami

## 🙏 Acknowledgments

- Nature Communications journal for publishing the original research
- UCI Machine Learning Repository for providing the dataset
- Scikit-learn, Pandas, and Flask communities for excellent tools
- Course instructor for guidance and support

## 📬 Contact

For questions, suggestions, or collaborations:
- Email: parinaztorabi2001@gmail.com

---

**Project Status**: Completed - January 2026  
**Last Updated**: January 2026  