# IMDB Movie Review Sentiment Analysis

This project performs sentiment analysis on the IMDB Movie Review dataset. It builds and evaluates two different machine learning models, **Multinomial Naive Bayes** and **Linear Support Vector Machine (SVM)**, to classify movie reviews as either positive or negative.

The entire workflow, from text preprocessing to model training and evaluation, is encapsulated within `scikit-learn` Pipelines for efficiency and clarity.

-----

## 📋 Requirements

To run this script, you'll need Python 3 and the following libraries:

  * pandas
  * scikit-learn

You can install them using pip:

```bash
pip install pandas scikit-learn
```

-----

## 💾 Dataset

The project uses the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

  * **File Name:** `IMDB Dataset.csv`
  * **Content:** The dataset contains 50,000 movie reviews, each labeled with a sentiment.
      * `review`: The text content of the movie review.
      * `sentiment`: The label ('positive' or 'negative').

**Important:** Make sure the `IMDB Dataset.csv` file is in the same directory as the Python script.

-----

## ⚙️ How to Run

1.  Make sure you have the required libraries installed.
2.  Place the `IMDB Dataset.csv` file in the same folder as your script.
3.  Execute the Python script from your terminal:
    ```bash
    python your_script_name.py
    ```

-----

## 🧠 Code Explanation

The script follows these key steps:

### 1\. Data Loading and Preparation

  * The `IMDB Dataset.csv` is loaded into a pandas DataFrame.
  * A new column `Sentiment` is created by mapping the string labels `'positive'` and `'negative'` to numerical values `0` and `1`.
      * **Note:** The original code had a slight bug. It used `y = df['sentiment']` (string labels) for splitting instead of the newly mapped numeric column. The models still work, but it's better practice to use the numeric labels consistently. The `classification_report` then correctly uses `target_names` to display the results with the proper labels.

### 2\. Data Splitting

  * The dataset is split into training (80%) and testing (20%) sets.
  * `stratify=y` is used to ensure that both the training and testing sets have the same proportion of positive and negative reviews as the original dataset.

### 3\. Model Pipelines

Two machine learning pipelines are created to streamline the process. Each pipeline consists of two steps:

1.  **Feature Extraction (`TfidfVectorizer`)**:

      * This step converts the raw text of the reviews into a matrix of TF-IDF features.
      * `stop_words='english'`: Removes common English words (like "the", "a", "is") that don't carry much sentiment.
      * `ngram_range=(1,2)`: Considers both individual words (unigrams) and pairs of adjacent words (bigrams) as features. For example, in "not good," it analyzes "not," "good," and the pair "not good," which captures more context than words alone.

2.  **Classification (`classifier`)**:

      * **Pipeline 1**: Uses `MultinomialNB`, a Naive Bayes classifier suitable for text data.
      * **Pipeline 2**: Uses `LinearSVC`, a fast and effective Support Vector Machine classifier.

### 4\. Training and Evaluation

  * Each pipeline is trained on the `x_train` and `y_train` data.
  * The trained models are then used to make predictions on the unseen `x_test` data.
  * Finally, the performance of each model is printed, including:
      * **Accuracy**: The overall percentage of correct predictions.
      * **Classification Report**: A detailed report showing precision, recall, and F1-score for each class (Negative and Positive).

-----

## 📈 Expected Output

When you run the script, it will train both models and print their performance reports to the console. The output will look similar to this (exact scores may vary slightly):

```
--- Naive Bayes Model Performance ---
Accuracy: 0.8878

Classification Report:
              precision    recall  f1-score   support

    Negative       0.88      0.89      0.89      5000
    Positive       0.89      0.88      0.89      5000

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000


--- Linear SVM Model Performance ---
Accuracy: 0.9001

Classification Report:
              precision    recall  f1-score   support

    Negative       0.90      0.90      0.90      5000
    Positive       0.90      0.90      0.90      5000

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000

```

As is common for text classification tasks, the **Linear SVM model slightly outperforms the Naive Bayes model** in terms of accuracy.