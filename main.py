import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#Load dataset
data = pd.read_csv("movies.csv")

#Features and labels
X = data["description"]
Y = data["genre"]

#Convert text to numbers
cv = CountVectorizer()
X = cv.fit_transform(X)

#Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 42)

#Train model
model = MultinomialNB()
model.fit(X_train, Y_train)

#Accuracy
print("Model Accuracy:", model.score(X_test, Y_test))

#Prediction
user_input = input("Enter movie description: ")
text = cv.transform([user_input])
print("Predicted Genre:", model.predict(text)[0])
