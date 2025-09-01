import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_csv('Reviews.csv')

# 2. Data Cleaning and Preparation
print("Preparing data...")
df.dropna(subset=['Text', 'Score'], inplace=True)
df_sample = df.sample(n=5000, random_state=42)
df_sample = df_sample[df_sample['Score'] != 3]
df_sample['Sentiment'] = df_sample['Score'].apply(lambda score: 'positive' if score > 3 else 'negative')

# 3. Define features (X) and target (y)
X = df_sample['Text']
y = df_sample['Sentiment']

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data preparation complete.")


# 5. Create a model pipeline
print("Training model...")
# A pipeline chains steps together: first, transform the text data, then classify it.
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(solver='liblinear'))
])

# 6. Train the model
# The .fit() command is where the model learns from the training data.
model.fit(X_train, y_train)

# 7. Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model training complete.")
print(f"Accuracy on test data: {accuracy:.4f}")

# 8. Save the trained model
joblib.dump(model, 'sentiment_model.pkl')
print("Model saved as sentiment_model.pkl")