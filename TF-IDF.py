import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('data/mt_train_label.csv')

# Drop NA values (like row 8446)
data = data.dropna()

# Combine the text fields for feature extraction
data['combined_text'] = data['hyp_normalized'] + " " + data['tgt_normalized']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['combined_text'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)  # Use n-grams (1, 2) as features
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_vec, y_train)

# Predict with Logistic Regression
y_pred_logistic = logistic_model.predict(X_test_vec)

# Evaluate Logistic Regression
print("Logistic Regression Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logistic)}")
print("Classification Report:")
print(classification_report(y_test, y_pred_logistic))

# Support Vector Machines
svm_model = SVC(kernel='linear', probability=True)  # Use a linear kernel for SVM
svm_model.fit(X_train_vec, y_train)

# Predict with SVM
y_pred_svm = svm_model.predict(X_test_vec)

# Evaluate SVM
print("\nSVM Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))
