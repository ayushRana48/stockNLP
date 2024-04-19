import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer 
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from joblib import dump, load
import ast

# set of stopwords from NLTK 
stops = set(stopwords.words('english'))

def preprocess(text):
    sentence_list = []
    for word in text.lower().split():
        if word not in stops:

            word_list = []
            for char in word:
                if char.isalpha():
                    word_list.append(char)
            if len(word_list) != 0:
                sentence_list.append(''.join(word_list))
    return ' '.join(sentence_list)

assert preprocess("A true random number generator (TRNG), also known as a hardware random number generator (HRNG), does not use a computer algorithm. Instead, it uses an external unpredictable physical variable such as radioactive decay of isotopes or airwave static to generate random numbers.") == 'true random number generator trng also known hardware random number generator hrng use computer algorithm instead uses external unpredictable physical variable radioactive decay isotopes airwave static generate random numbers'
assert preprocess("This IS A test !!!! I hope this makes SENSE") == 'test hope makes sense'

df1 = pd.read_csv('paragraph.csv')

# drop irrelevant columns and nulls
df = df1.drop(columns=['Unnamed: 0', 'ticker', 'link'])
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# apply preprocessing and create boolean columns with numerical val 
df['relevant'] = df['relevant'].apply(lambda x: 1 if x == True else 0)
# df['paragraph'] = df['paragraph'].apply(lambda x: preprocess(x))
df

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df.paragraph)
y = df.relevant

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

param_grid = {
    'C': np.logspace(-4, 4, 50),  # Regularization strength
    'penalty': ['l1', 'l2'],  # Types of regularization
    'solver': ['liblinear']  # Solver that supports l1 penalty
}

# Initialize the model with solver that supports l1
model = LogisticRegression(solver='liblinear')

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, verbose=True, n_jobs=-1)

# Fit model on training data
model = grid_search.fit(X_train_smote, y_train_smote);

y_pred = model.predict(X_test)

def verify_logreg(model, lst, vectorizer):

    preprocessed_list = [preprocess(para) for para in lst]
    
    # Transform all preprocessed paragraphs using the already fitted vectorizer
    X = vectorizer.transform(preprocessed_list)

    # Predict using the logistic regression model
    y_pred = model.predict(X)
    
    # Append paragraphs where the prediction is 0
    new = [para for idx, para in enumerate(lst) if y_pred[idx] != 0]
    
    # Return the combined text of the filtered paragraphs
    return ' '.join(new)

preprocLst1 = ["hello want", "hello kitty", "talk about real life"]
assert verify_logreg(model, preprocLst1, vectorizer) == 'hello want talk about real life'
preprocLst2 = ["world burn alive hot", "environmental change"]
assert verify_logreg(model, preprocLst2, vectorizer) == "environmental change"

print("Success!")