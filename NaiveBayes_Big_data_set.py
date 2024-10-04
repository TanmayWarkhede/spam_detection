import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset
df_raw = pd.read_csv('enron_spam_data.csv', dtype=str, low_memory=False)

# Clean the data
df_raw = df_raw.drop(columns=[x for x in df_raw.columns if x.startswith('Unnamed')])
df_raw["Spam/Ham"] = df_raw['Spam/Ham'].apply(lambda x: 1 if x == "spam" else 0)
df_raw['Combined'] = df_raw['Subject'] + " " + df_raw['Message']
df_raw = df_raw.dropna(subset=['Combined'])

# Split the data
input_label = df_raw['Combined']
output_label = df_raw['Spam/Ham']
input_train, input_test, output_train, output_test = train_test_split(input_label, output_label, test_size=0.2)

# Vectorize the data
cv = CountVectorizer()
input_train_vocabulary = cv.fit_transform(input_train)

# Train the model
model = MultinomialNB()
model.fit(input_train_vocabulary, output_train)

# Save the model and vectorizer
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(cv, "vectorizer.pkl")

# Evaluate the model
input_test_vocabulary = cv.transform(input_test)
accuracy = model.score(input_test_vocabulary, output_test)
print(f'Accuracy: {accuracy}')
