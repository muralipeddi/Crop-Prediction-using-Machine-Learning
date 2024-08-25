import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
crop = pd.read_csv('Crop_recommendation.csv') 
crop.head()
crop.shape 
crop.isnull().sum() 
crop.duplicated().sum()
crop.info()
crop.describe()
grouped = crop.groupby("label") 
grouped.mean()["N"].plot(kind="barh") 
grouped.mean()["P"].plot(kind="barh") 
grouped.mean()["K"].plot(kind="barh") 
grouped.mean()["temperature"].plot(kind="barh") 
grouped.mean()["rainfall"].plot(kind="barh") 
grouped.mean()["humidity"].plot(kind="barh") 
grouped.mean()["ph"].plot(kind="barh") 
crop['label'].value_counts()
crop_dict = { 'rice': 1,'maize': 2,'jute': 3,'cotton': 4,'coconut': 5, 'papaya': 6,'orange': 7,'apple': 8,'muskmelon': 9, 'watermelon': 10,'grapes': 11,'mango': 12,'banana': 13, 'pomegranate': 14,'lentil': 15,'blackgram': 16, 'mungbean': 17,'mothbeans': 18,'pigeonpeas': 19, 'kidneybeans': 20,'chickpea': 21, 'coffee': 22 }
crop['label_num'] = crop['label'].map(crop_dict) 
crop.drop('label',axis=1,inplace=True)
crop.head()

# Split the dataset into features and labels
X = crop.iloc[:, :-1]
y = crop.iloc[:, -1]
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # create instances of all models
models = { 'Logistic Regression': LogisticRegression(), 'Naive Bayes': GaussianNB(), 'Support Vector Machine': SVC(), 'K-Nearest Neighbors': KNeighborsClassifier(), 'Decision Tree': DecisionTreeClassifier(), 'Random Forest': RandomForestClassifier(), }
from sklearn.metrics import accuracy_score 
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) 
    print(f'{name}:\nAccuracy: {acc:.4f}')

# Selecting decistion tree model:
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test) 
print(accuracy_score(y_test,y_pred))

# Define function to make predictions
def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    # Create a numpy array with the input values
    input_values = np.array([[N, P, K, temperature, humidity, pH, rainfall]]) # Use the model to make a prediction
    prediction = rfc.predict(input_values) # Return the predicted crop label
    return prediction[0]
    N = 114
    P = 21
    K = 55
    tem = 25.44 
    humidity = 87.94 
    ph = 6.47 
    rainfall = 257.52
    pred = predict_crop(N, P, K, tem, humidity, ph, rainfall)
    if pred == 1:
        print("Rice is the best crop to be cultivated right there")
    elif pred == 2:
        print("Maize is the best crop to be cultivated right there")
    elif pred == 3:
        print("Jute is the best crop to be cultivated right there")
    elif pred == 4:
        print("Cotton is the best crop to be cultivated right there")
    elif pred == 5:
        print("Coconut is the best crop to be cultivated right there") 
    elif pred == 6:
        print("Papaya is the best crop to be cultivated right there")
    elif pred == 7:
        print("Orange is the best crop to be cultivated right there")
    elif pred == 8:
        print("Apple is the best crop to be cultivated right there")
    elif pred == 9:
        print("Muskmelon is the best crop to be cultivated right there")
    elif pred == 10:
        print("Watermelon is the best crop to be cultivated right there")
    elif pred == 11:
        print("Grapes is the best crop to be cultivated right there")
    elif pred == 12:
        print("Mango is the best crop to be cultivated right there")
    elif pred == 13:
            print("Banana is the best crop to be cultivated right there")
    elif pred == 14:
        print("Pomegranate is the best crop to be cultivated right there") 
    elif pred == 15:
        print("Lentil is the best crop to be cultivated right there") 
    elif pred == 16:
        print("Blackgram is the best crop to be cultivated right there")
    elif pred == 17:
        print("Mungbean is the best crop to be cultivated right there") 
    elif pred == 18:
        print("Mothbeans is the best crop to be cultivated right there")
    elif pred == 19:
        print("Pigeonpeas is the best crop to be cultivated right there") 
    elif pred == 20:
        print("Kidneybeans is the best crop to be cultivated right there")
    elif pred == 21:
        print("Chickpea is the best crop to be cultivated right there") 
    elif pred == 22:
        print("Coffee is the best crop to be cultivated right there") 
    else:
        print("Sorry, we could not determine the best crop to be cultivated with the provided data.")
import pickle
pickle.dump(rfc, open('model.pkl','wb')) 
X_train