import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Improved dataset
data = {
    "text": [
        "Government announces new policy",
        "Scientists discover new planet",
        "Elections results are out",
        "New education reforms introduced",
        "Stock market hits record high",
        "India launches new satellite",

        "Win money instantly by clicking this link",
        "Click here to earn 5000 per day",
        "You won a lottery claim now",
        "Breaking shocking scandal you won't believe",
        "Fake news spreading on social media",
        "Earn money fast with this trick"
    ],
    "label": [
        1, 1, 1, 1, 1, 1,   # Real
        0, 0, 0, 0, 0, 0    # Fake
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split data
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# User input loop
print("\n--- Fake News Detector ---")

while True:
    news = input("\nEnter news text (or type 'exit' to quit): ")

    if news.lower() == "exit":
        print("Exiting program...")
        break

    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)

    if prediction[0] == 1:
        print("✅ This news is REAL")
    else:
        print("❌ This news is FAKE")