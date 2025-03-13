from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data-source\spambase.csv')
X = df.drop('spam', axis=1)
y = df['spam']

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print('Modelo salvo como "spam_model.pkl"')
