# Human Activity Recognition using a Hybrid Deep Learning Model

This project focuses on Human Activity Recognition (HAR) using wearable inertial sensor data and a hybrid deep learning architecture. The goal is to accurately classify human activities by combining convolutional, recurrent, and attention-based models to capture both spatial and temporal patterns in multivariate sensor signals.

The work was carried out as part of an academic research project at the University of Siegen, with an emphasis on building reproducible machine learning pipelines and evaluating model robustness on real-world sensor data.


## Dataset

The dataset consists of multivariate inertial sensor signals collected from wearable devices. Each sample includes time-series measurements from accelerometer and gyroscope sensors recorded across multiple subjects performing daily activities.

The raw sensor data is segmented using a sliding window approach and processed to extract meaningful temporal patterns for model training and evaluation. This setup reflects realistic conditions encountered in wearable sensing and activity recognition tasks.


🔍 Problem Overview

Human Activity Recognition from wearable sensors is a key problem in:

Healthcare monitoring

Affective computing

Human–machine interaction

Smart environments

The goal is to accurately classify human activities from time-series inertial sensor data while handling temporal complexity and sensor noise.

🧠 Model Architecture

The proposed hybrid deep learning architecture integrates multiple modeling paradigms to effectively capture spatial, temporal, and contextual dependencies in wearable sensor data:

- Convolutional Neural Networks (CNNs) for local feature extraction from segmented time-series windows  
- Bidirectional LSTM and GRU layers for modeling short- and long-term temporal dependencies  
- Transformer Encoder layers to capture global context through attention mechanisms  

This hybrid design enables improved representation learning compared to traditional CNN–LSTM baselines by combining recurrent dynamics with attention-based global reasoning.


📊 Results

- Accuracy: **98.56%**  
- F1-Score: **95.94%**  

The model demonstrates strong generalization across subjects and improved robustness due to effective sensor fusion and the hybrid architectural design.


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
## Limitations

- The model was evaluated on a limited number of subjects and activities.
- Performance may vary under different sensor configurations or sampling rates.
- Real-time deployment constraints were not evaluated in this study.

## Future Work

Future extensions of this work may include real-time deployment on edge devices, evaluation on larger multi-modal datasets, and integration with self-supervised or foundation models for improved generalization.


📚 Context

This project was developed as part of academic research and reflects applied machine learning and deep learning practices relevant to AI Engineer, ML Engineer, and Research Assistant roles.

## Reproducibility Notes

All experiments were conducted in a Linux-based environment using fixed random seeds to ensure reproducibility. Model training, evaluation, and visualization steps are fully documented within the provided Jupyter notebook.

📬 Contact

Srinadh Alugu
AI / ML Engineer
srinadhalugu@gmail.com
