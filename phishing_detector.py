import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("emails.csv")

# Features and labels
X = data['text']
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert text into vectors
vectorizer = TfidfVectorizer(stop_words='english')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, predictions))

# Predict custom emails
while True:
    msg = input("\nEnter email text: ")

    transformed = vectorizer.transform([msg])
    prediction = model.predict(transformed)

    if prediction[0] == 1:
        print("⚠ Phishing Email")
    else:
        print("✓ Legitimate Email")
