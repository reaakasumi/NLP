import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE



def model_nb(df_train, df_val, df_test): 
    X_train = df_train.drop(columns = ['label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns = ['label'])
    y_test = df_test['label']
    X_val = df_val.drop(columns = ['label'])
    y_val = df_val['label']
    X_val['combined_text'] = X_val['hyp_normalized'] + " " + X_val['tgt_normalized']
    X_train['combined_text'] = X_train['hyp_normalized'] + " " + X_train['tgt_normalized']
    X_test['combined_text'] = X_test['hyp_normalized'] + " " + X_test['tgt_normalized']
    X_val = X_val['combined_text'].fillna('').astype(str)
    X_train = X_train['combined_text'].fillna('').astype(str)
    X_test = X_test['combined_text'].fillna('').astype(str)
    """Perform hyperparameter tuning using GridSearchCV."""
    #here we need X_train and y_train 
    # Define the pipeline for preprocessing and model training
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),  # Vectorizer with stopword removal
        ('nb', MultinomialNB())       # Naive Bayes classifier
    ])

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'tfidf__max_features': [2000, 3000, 5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        #'tfidf__min_df': [0,1, 2, 3],
        'nb__alpha': [1.0, 0.5, 0.1, 0.01]
    }

    # Perform grid search with stratified cross-validation
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words=stopwords.words('english'))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

    # Evaluate the best model on the validation set
    y_val_pred = best_model.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # Retrain the best model on the full training data (including validation)
    best_model.fit(X_train, y_train)

    # Evaluate the retrained model on the test set
    y_test_pred = best_model.predict(X_test)
    result = pd.DataFrame({'label': y_test, 'prediction_nb': y_test_pred})
    return(result)

#load mt data
df_mt_train = pd.read_csv('data/preprocessed/mt_train_label.csv')
df_mt_val = pd.read_csv('data/preprocessed/mt_val_label.csv')
df_mt_test = pd.read_csv('data/preprocessed/mt_test_label.csv')

#load dm data
df_dm_train = pd.read_csv('data/preprocessed/dm_train_label.csv')
df_dm_val = pd.read_csv('data/preprocessed/dm_val_label.csv')
df_dm_test = pd.read_csv('data/preprocessed/dm_test_label.csv')

#load pg data
df_pg_train = pd.read_csv('data/preprocessed/pg_train_label.csv')
df_pg_val = pd.read_csv('data/preprocessed/pg_val_label.csv')
df_pg_test = pd.read_csv('data/preprocessed/pg_test_label.csv')

df_pg_train = df_pg_train.rename(columns = {'src_normalized': 'tgt_normalized'})
df_pg_val = df_pg_val.rename(columns = {'src_normalized': 'tgt_normalized'})
df_pg_test = df_pg_test.rename(columns = {'src_normalized': 'tgt_normalized'})

print('Results for MT')
data_result_mt = model_nb(df_mt_train, df_mt_val, df_mt_test)
print('Results for DM')
data_result_dm = model_nb(df_dm_train,df_dm_val, df_dm_test)
print('Results for PG')
data_result_pg = model_nb(df_pg_train,df_pg_val, df_pg_test)
    
    
    
    
    
    
