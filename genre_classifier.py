import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

#Configuration 
REMOVE_RARE_GENRES = True
RARE_GENRE_THRESHOLD = 10

# Load training data (genre + plot)
def load_training_data(path):
    genres, plots = [], []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if len(parts) >= 4:
                genres.append(parts[2].strip().lower())  
                plots.append(parts[3].strip())
    if not genres:
        raise ValueError(f"No valid data found in file: {path}")
    return pd.DataFrame({'genre': genres, 'plot': plots})

# Load test data (only plots)
def load_test_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")
    with open(path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

# Dataset paths
train_path = os.path.join("Genre Classification Dataset", "train_data.txt")
test_path = os.path.join("Genre Classification Dataset", "test_data.txt")
solution_path = os.path.join("Genre Classification Dataset", "test_data_solution.txt")

# Load data
df = load_training_data(train_path)
test_plots = load_test_data(test_path)

# Optionally remove rare genres
if REMOVE_RARE_GENRES:
    genre_counts = df['genre'].value_counts()
    df = df[df['genre'].isin(genre_counts[genre_counts >= RARE_GENRE_THRESHOLD].index)]

print("Sample training data:")
print(df.head())
print("Training data shape:", df.shape)

# Stratified split to maintain genre distribution
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in split.split(df['plot'], df['genre']):
    X_train, X_val = df['plot'].iloc[train_idx], df['plot'].iloc[val_idx]
    y_train, y_val = df['genre'].iloc[train_idx], df['genre'].iloc[val_idx]

# Model pipeline with enhancements
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))),
    ('lr', LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight='balanced'))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate on validation data
val_preds = model.predict(X_val)
print("\nValidation Results:\n")
print(classification_report(y_val, val_preds, zero_division=0))

# Predict genres for test data
test_preds = model.predict(test_plots)
test_preds_lower = [pred.lower() for pred in test_preds]  # lowercase predictions

# Save predictions
with open('test_predictions.txt', 'w', encoding='utf-8') as f:
    for pred in test_preds_lower:
        f.write(pred + '\n')

# Optional: Evaluate against true test labels
if os.path.exists(solution_path):
    true_labels = []
    with open(solution_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if len(parts) >= 4:
                true_labels.append(parts[2].strip().lower())  # Extract genre and lowercase

    if len(true_labels) != len(test_preds_lower):
        print("Warning: Mismatch in true and predicted label counts.")
        print(f"True labels: {len(true_labels)}, Predictions: {len(test_preds_lower)}")
    else:
        unique_true = set(true_labels)
        unique_pred = set(test_preds_lower)

        print(f"Unique true labels: {len(unique_true)}")
        print(f"Unique predicted labels: {len(unique_pred)}")

        if len(unique_true) > 50:
            acc = accuracy_score(true_labels, test_preds_lower)
            print(f"\nTest Accuracy: {acc:.4f}")
            print("Too many unique classes for detailed classification report, skipping it.")
        else:
            print("\nTest Set Evaluation:\n")
            print(classification_report(true_labels, test_preds_lower, zero_division=0))
else:
    print("Note: test_data_solution.txt not found. Skipping test set evaluation.")
