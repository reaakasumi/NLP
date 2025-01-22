import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

df_train = pd.read_csv("data/generated/train_features.csv")
df_test = pd.read_csv("data/generated/test_features.csv")

vectorizer = CountVectorizer(max_features=500)  # Limit features to 500 most frequent words
reference_vectors_train = vectorizer.fit_transform(df_train['tgt_src']).toarray()
hypothesis_vectors_train = vectorizer.transform(df_train['hyp']).toarray()

reference_vectors_test = vectorizer.transform(df_test['tgt_src']).toarray()
hypothesis_vectors_test = vectorizer.transform(df_test['hyp']).toarray()


# training data
X_train = df_train[['overlap', 'semantic_similarity', 'bleu_score', 'entity_overlap']]
X_train = np.hstack([X_train.values, reference_vectors_train, hypothesis_vectors_train])
y_train = df_train['label']


# testing data
X_test = df_test[['overlap', 'semantic_similarity', 'bleu_score', 'entity_overlap']]
X_test = np.hstack([X_test.values, reference_vectors_test, hypothesis_vectors_test])
y_test = df_test['label']

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# classifier
clf = RandomForestClassifier()
clf.fit(X_train_resampled, y_train_resampled)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


print(classification_report(y_test, y_pred))

# save predicted data
df_test["predicted_fc"] = y_pred
df_test.to_csv("data/generated/test_labeled_fc.csv")
