# Spam Message Detection Project

A machine learning project that classifies SMS messages as spam or ham (legitimate) using various classification algorithms, natural language processing techniques, and deep learning approaches.

## Overview

This project implements a comprehensive spam detection system that:
- Preprocesses and cleans SMS message data
- Performs exploratory data analysis (EDA)
- Engineers features from text data
- Trains and compares multiple machine learning models (traditional ML and deep learning)
- Evaluates model performance using accuracy, precision, recall, and F1-score metrics

## Dataset

The project uses a spam SMS dataset (`spam.csv`) containing labeled messages. The dataset includes:
- **Label**: Classification as spam (1) or ham (0)
- **Message**: The SMS text content

After cleaning, the dataset contains **5,169 messages** (duplicates removed).

## Features

### Data Preprocessing
- Encoding detection and handling
- Duplicate removal
- Label encoding
- Text transformation:
  - Lowercasing
  - Tokenization
  - Special character removal
  - Stop word removal
  - Stemming using Porter Stemmer

### Feature Engineering
- Basic features: character count, word count, sentence count
- Advanced features:
  - Message length
  - Word count
  - Average word length
  - Capital ratio
  - Punctuation count

### Machine Learning Models
The project compares 11 different traditional classification algorithms plus 1 deep learning model:

#### Traditional ML Models:
1. **Extra Trees Classifier** - Accuracy: 97.78%, Precision: 96.75%
2. **Random Forest** - Accuracy: 97.68%, Precision: 97.50%
3. **SVC** (Support Vector Classifier) - Accuracy: 97.58%, Precision: 97.48%
4. **Multinomial Naive Bayes** - Accuracy: 97.10%, Precision: 100%
5. **XGBoost** - Accuracy: 97.10%, Precision: 95.00%
6. **Logistic Regression** - Accuracy: 95.65%, Precision: 96.97%
7. **Bagging Classifier** - Accuracy: 95.94%, Precision: 86.92%
8. **Gradient Boosting** - Accuracy: 95.07%, Precision: 93.07%
9. **Decision Tree** - Accuracy: 93.23%, Precision: 83.33%
10. **AdaBoost** - Accuracy: 92.36%, Precision: 83.91%
11. **K-Nearest Neighbors** - Accuracy: 90.52%, Precision: 100%

#### Deep Learning Model:
12. **LSTM Neural Network** - Accuracy: 98.45%, Precision: 96.40%, Recall: 92.41%, F1-Score: 94.39%
   - Architecture: Embedding → LSTM (64 units) → Dense layers → Sigmoid output
   - Uses class weights to handle imbalanced data
   - Implements early stopping to prevent overfitting
   - Optimal threshold: 0.4 (Precision: 93.06%, Recall: 92.41%, F1: 93.38%)

### Vectorization
- **TF-IDF Vectorizer**: Term Frequency-Inverse Document Frequency (max_features=3000)
- **Count Vectorizer**: Bag-of-words approach (max_features=3000)
- **Deep Learning**: Tokenizer with sequence padding (max_words=5000, max_len=100)

## Installation

1. Clone or download this repository


2. Create a virtual environment (recommended):
```bash
python3.12 -m venv venv_3.12
source venv_3.12/bin/activate  # On macOS/Linux
# or
venv_3.12\Scripts\activate  # On Windows
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Download NLTK data (run in Python or Jupyter):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

1. Place your `spam.csv` file in the project directory
2. Open `sms_classifier.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially

The notebook includes:
- Data loading and cleaning
- Exploratory data analysis with visualizations
- Text preprocessing pipeline
- Feature engineering
- Traditional ML model training and evaluation
- Deep learning model (LSTM) training and evaluation
- Performance comparison visualizations
- Threshold analysis for optimal precision/recall balance

## Project Structure

```
.
├── Untitled6.ipynb                    # Main Jupyter notebook with analysis
├── spam.csv                           # Dataset file (not included)
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## Key Results

- **Best Overall Model**: LSTM Neural Network (98.45% accuracy, 96.40% precision, 92.41% recall)
- **Best Traditional ML Model**: Extra Trees Classifier (97.78% accuracy, 96.75% precision)
- **Best Precision**: Multinomial Naive Bayes and KNN (100% precision)
- **Best Balanced Performance**: Random Forest (97.68% accuracy, 97.50% precision)

### Deep Learning Performance (LSTM)
- **Test Accuracy**: 98.45%
- **Test Precision**: 96.40%
- **Test Recall**: 92.41%
- **F1-Score**: 94.39% (at threshold 0.5)
- **Optimal Threshold**: 0.4 (best balance: 93.06% precision, 92.41% recall, 93.38% F1)

All models show strong performance (>90% accuracy), indicating the effectiveness of the preprocessing and feature engineering pipeline. The deep learning approach demonstrates superior performance with better recall compared to traditional ML models.

## Visualizations

The project includes:
- Distribution plots for spam vs ham messages
- Pair plots showing feature relationships
- Correlation heatmaps
- Word clouds for spam messages
- Model performance comparison bar charts

## Technologies Used

- **Python 3.12**
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and utilities
- **TensorFlow/Keras**: Deep learning framework
- **NLTK**: Natural language processing
- **Matplotlib/Seaborn**: Data visualization
- **WordCloud**: Text visualization
- **XGBoost**: Gradient boosting framework

## Future Improvements

- [x] Deep learning model implementation (LSTM)
- [x] Comprehensive evaluation metrics (recall, F1-score)
- [x] Class weight handling for imbalanced data
- [ ] Model persistence (save/load trained models)
- [ ] Hyperparameter tuning for best models
- [ ] Cross-validation implementation
- [ ] Web application interface for predictions
- [ ] Real-time prediction API
- [ ] Enhanced feature engineering (n-grams, sentiment analysis)
- [ ] Experiment with transformer models (BERT, DistilBERT)
