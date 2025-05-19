# Fake_instagram_account_detection

This project focuses on detecting fake Instagram accounts using a combination of numerical and textual features. The system integrates modern NLP techniques, structured data analysis, and an ensemble learning approach to achieve high prediction accuracy.

---

## Project Overview

The pipeline includes:
- Data scraping
- Feature engineering
- Training multiple models on different data modalities
- Combining them into a final ensemble model for improved performance

---

## Dataset Setup

We utilized two datasets:

- **Small Custom Dataset**: Manually scraped data containing usernames, bios, etc. Primarily used for training NLP-based models.
- **Larger Structured Dataset**: Contains numerical features like follower count, following count, post count, privacy status, and more. Used for training the numerical model.

---

##  Approach

### 1. **Textual Input Models (Small Dataset)**
- **BERT (Fine-tuned)**  
  - F1 Score: **0.988**  
  - Accuracy: **98.8%**

- **TF-IDF + Classifier**  
  - F1 Score: **0.84**  
  - ROC AUC: **0.95**

- **BERT + Random Forest (Feature Extractor)**  
  - F1 Score: **0.96**  
  - ROC AUC: **0.994**

*The best-performing NLP model was selected for ensemble integration.*

### 2. **Numerical Model (Large Dataset)**
- Keras-based neural network trained on structured features such as:
  - Follower count
  - Following count
  - Number of posts
  - Privacy status
  - Other engineered features (digit ratio in username, bio length, etc.)

### 3. **Final Ensemble Model**
- Combines predictions from:
  - Best NLP model
  - Numerical model
- Uses **Logistic Regression** as a meta-classifier.

---

##  Final Ensemble Performance (on Held-Out Test Set)

| Metric       | Score   |
|--------------|---------|
| Accuracy     | 96.5%   |
| Precision    | 94.7%   |
| Recall       | 98.6%   |
| F1 Score     | 96.6%   |
| ROC AUC      | 0.995   |

---

##  Deployment

You can run the final system via:

1. **Instagram Username**: Automatically scraped using Instaloader.
2. **Manual Input**: For testing synthetic or real-looking profiles.

###  Output
- Binary classification: **Fake** or **Real**
- Includes a **confidence score** for prediction certainty

---

##  Required Files

Ensure the following files are in the working directory before running the deployment:

- `keras_model_wrapper1.pkl` – Trained numerical model
- `bert_model.pkl` – Trained BERT-based NLP model  
  *(Generate this via `finetune_bert.ipynb`)*
- `numerical_scaler.pkl` – Scaler for preprocessing structured features
- `meta_model.pkl` – Logistic regression meta-model for ensemble prediction

---

## How to Use

1. **Install Required Packages**
   ```bash
   pip install instaloader pandas joblib torch transformers scikit-learn keras
2. Run the deployment script (ensemble.ipynb or equivalent .py file).

3. Enter either Instagram usernames or manually crafted profile inputs.

4. The script will handle scraping, preprocessing, feature extraction, and prediction.

5. The Instagram scraping tool is rate-limited. To prevent being blocked, we’ve included a wait/sleep logic in case Instagram starts throttling.

 If the username doesn’t exist, the script handles the error and skips that user.
Feature engineering steps (like digit ratio, bio length) are automatically applied during prediction time.
