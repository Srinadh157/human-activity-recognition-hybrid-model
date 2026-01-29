# Human Activity Recognition using a Hybrid Deep Learning Model

This project focuses on Human Activity Recognition (HAR) using wearable inertial sensor data and a hybrid deep learning architecture. The goal is to accurately classify human activities by combining convolutional, recurrent, and attention-based models to capture both spatial and temporal patterns in multivariate sensor signals.

The work was carried out as part of an academic research project at the University of Siegen, with an emphasis on building reproducible machine learning pipelines and evaluating model robustness on real-world sensor data.

🔍 Problem Overview

Human Activity Recognition from wearable sensors is a key problem in:

Healthcare monitoring

Affective computing

Human–machine interaction

Smart environments

The goal is to accurately classify human activities from time-series inertial sensor data while handling temporal complexity and sensor noise.

🧠 Model Architecture

The proposed hybrid architecture consists of:

CNN layers for local feature extraction

BiLSTM and GRU layers for sequential and temporal modeling

Transformer Encoder for attention-based global context learning

This combination outperforms traditional CNN–LSTM baselines by leveraging both recurrent and attention mechanisms.

📊 Results

Accuracy: 98.56%

F1-Score: 95.94%

Demonstrated strong generalization across subjects

Improved robustness through sensor fusion and architectural design

📁 Repository Structure
human-activity-recognition-hybrid-model/
│
├── Data/                 # Wearable sensor datasets (CSV files)
├── notebooks/            # Jupyter notebook with model implementation
│   └── har_hybrid_model.ipynb
├── results/              # Evaluation reports and documentation
│   └── Group-3_RAML_Report.pdf
├── README.md             # Project documentation

⚙️ Technologies Used

Python

PyTorch

NumPy, Pandas

Scikit-learn

Matplotlib / Seaborn

Jupyter Notebook

Linux-based experimentation

🚀 How to Run

Clone the repository:

git clone https://github.com/Srinadh157/human-activity-recognition-hybrid-model.git
cd human-activity-recognition-hybrid-model


Open the notebook:

jupyter notebook notebooks/har_hybrid_model.ipynb


Run all cells to:

Load data

Train the hybrid model

Evaluate performance metrics

📌 Key Highlights

End-to-end ML pipeline: preprocessing → modeling → evaluation

Sliding-window segmentation and feature normalization

Comparative evaluation against baseline models

Focus on robustness and real-world applicability

📚 Context

This project was developed as part of academic research and reflects applied machine learning and deep learning practices relevant to AI Engineer, ML Engineer, and Research Assistant roles.


📬 Contact

Srinadh Alugu
AI / ML Engineer
srinadhalugu@gmail.com
