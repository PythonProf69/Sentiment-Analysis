import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report

df=pd.read_csv('IMDB Dataset.csv')
df['Sentiment']=df['sentiment'].map({'positive':0,'negative':1})
x=df['review']
y=df['sentiment']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2,stratify=y)

nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('classifier', MultinomialNB())
])
nb_pipeline.fit(x_train,y_train)
y_pred_nb = nb_pipeline.predict(x_test)

print("\n--- Naive Bayes Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb, target_names=['Negative', 'Positive']))

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('classifier', LinearSVC(random_state=42))
])

svm_pipeline.fit(x_train, y_train)
y_pred_svm = svm_pipeline.predict(x_test)

print("\n--- Linear SVM Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['Negative', 'Positive']))