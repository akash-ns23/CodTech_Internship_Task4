import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Create a balanced dataset (3 spam, 3 ham)
data = {
    'label': ['ham', 'spam', 'ham', 'ham', 'spam', 'spam'],
    'message': [
        'Hey, are we still meeting today?',
        'You have won a free lottery ticket! Claim now.',
        'Don’t forget to submit your homework.',
        'Let’s grab coffee this weekend.',
        'URGENT: Your account has been compromised. Reset now!',
        'Earn $1000/day working from home!'
    ]
}

# Convert to DataFrame and map labels to numeric
df = pd.DataFrame(data)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 2: Convert text to numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

# Step 3: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Print evaluation
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
