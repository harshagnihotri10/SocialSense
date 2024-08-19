# SocialSense: Social Media Sentiment Analysis

SocialSense is a powerful machine learning model designed for sentiment analysis of social media posts. It classifies the sentiments (positive, negative, or neutral) expressed in posts, helping organizations gain valuable insights into public opinion. By using advanced Natural Language Processing (NLP) and machine learning algorithms, this model allows businesses to make data-driven decisions based on public sentiment around their brand, products, or events.

# Images:

![image](https://github.com/user-attachments/assets/9bff01d0-6b1e-430e-9254-221defe6d8ec)

![image](https://github.com/user-attachments/assets/8637a2cd-062a-43b2-8055-7b828e03e513)

![image](https://github.com/user-attachments/assets/7e8d2909-911f-49b1-92b6-715365e9820f)

![image](https://github.com/user-attachments/assets/4d242afd-8208-4087-97ae-540a3c2720f7)





## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)


## Features
- Sentiment classification (positive, negative, neutral) using machine learning.
- Data preprocessing with Natural Language Processing (NLP).
- Model training and evaluation using labeled social media datasets.
- Easy integration into existing applications for sentiment analysis.

## Tech Stack
- **Languages**: Python
- **Machine Learning Libraries**: Scikit-learn, Pandas, NumPy
- **Natural Language Processing**: NLTK, SpaCy
- **Model Deployment**: Flask (for API integration)
- **Dataset Handling**: CSV files


## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/harshagnihotri10/SocialSense.git
cd SocialSense
```

### 2. Install Dependencies
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Dataset Setup
Make sure you have the `train.csv` and `test.csv` datasets in the root directory.

## Usage

### 1. Data Preprocessing
Run the `Data Preprocessing.py` script to clean and preprocess the data:
```bash
python Data\ Preprocessing.py
```

### 2. Model Training
Use the preprocessed `train.csv` data to train the machine learning model:
```bash
# Train the model
python train_model.py
```

### 3. Model Evaluation
After training, evaluate the model's performance using `test.csv`:
```bash
# Evaluate the model
python evaluate_model.py
```

### 4. Sentiment Prediction
Use the trained model to classify sentiments from new social media posts by running the sentiment analysis prediction:
```bash
python predict_sentiment.py --input "Your social media post here"
```

## Datasets
- **train.csv**: Training data for sentiment analysis.
- **test.csv**: Test data for evaluating the model.

## Directory Structure
```
SocialSense/
│
├── Data Preprocessing.py       # Data preprocessing script (NLP, cleaning, feature extraction)
├── train_model.py              # Script for training the sentiment analysis model
├── evaluate_model.py           # Script to evaluate the trained model
├── predict_sentiment.py        # Script for making sentiment predictions
├── train.csv                   # Training dataset
├── test.csv                    # Test dataset
├── submission.csv              # Submission results file
├── README.md                   # Project documentation
├── requirements.txt            # List of dependencies for the project
└── SocialSense.pptx            # Project presentation
```

## Contributing
Contributions are welcome! If you’d like to improve the project, please fork the repository and submit a pull request. Make sure to run tests before submitting changes.
