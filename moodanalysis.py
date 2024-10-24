import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('sentimentdataset.csv')
X = data['Text']
y = data['Sentiment']

# Text preprocessing and vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Model building
model = MultinomialNB()
model.fit(X_train, y_train)

# Get user input
user_input = input("Enter a sentence: ")

# Preprocess and predict
user_input_tfidf = tfidf.transform([user_input])
prediction = model.predict(user_input_tfidf)
#prediction_proba = model.predict_proba(user_input_tfidf)
prediction_proba = model.predict_proba(user_input_tfidf)[0]
print(model.classes_)
# Print classification
print(f"The sentiment of the text is: {prediction[0]}")

# Visualize the result
#labels = ['Negative', 'Neutral', 'Positive']
#plt.pie(prediction_proba[0], labels=labels, autopct='%1.1f%%', colors=['red', 'yellow', 'green'])
#plt.title(f"Mood Classification for: '{user_input}'")
#plt.show()

# Custom function to display percentages above 5%
def autopct_format(values):
    def custom_autopct(pct):
        return ('%1.1f%%' % pct) if pct > 7 else ''
    return custom_autopct

# Get the classes the model was trained on
labels = model.classes_  # Dynamically fetch the class labels
def filter_below_5(labels, probs):
    new_labels = []
    new_probs = []
    for i in range(len(probs)):
        if probs[i]  >= 0.05:
            new_labels.append(labels[i])
            new_probs.append(probs[i])
    return new_labels, new_probs

# Filter out labels and probabilities below 5%
labels = model.classes_
filtered_labels, filtered_probs = filter_below_5(labels, prediction_proba)
# Visualize the result
plt.pie(prediction_proba,  autopct='%1.1f%%', colors=['red', 'yellow', 'green'][:len(labels)])
plt.title(f"Mood Classification for: '{user_input}'")
plt.show()