import pandas as pd
import pickle
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def fix_and_preprocess(csv_path):
    print("Loading dataset (this might take a while)...")
    df = pd.read_csv(r"C:\Users\Vvillanueva\Downloads\ud120-projects-master\ud120-projects-master\emails.csv")
    # FILTER ONLY AUTHORS THE COURSE NEEDS: SARA (1) AND CHRIS(0)
    sara_emails = df[df['file'].str.contains('shackleton-s', case=False)]
    chris_emails = df[df['file'].str.contains('germany-c', case=False)]
    
    # labeling Sara = 1, Chris = 0
    # change in order to follow course criteria
    # our kaggle dataset is different from the original ds
    sara_emails['label'] = 0   # Sara now must be 0
    chris_emails['label'] = 1  # Chris now must be 1
    
    combined_df = pd.concat([sara_emails, chris_emails])
    print(f"Emails found - Sara: {len(sara_emails)}, Chris: {len(chris_emails)}")

    # 2. TEXT CLEANING
    def clean_email(text):
        # KAGGLE CSV includes headers (To, From, Subject...).
        # Look the end of the heard 
        parts = text.split('\n\n', 1)
        content = parts[1] if len(parts) > 1 else parts[0]
        # Delete "SIGNATURES" or possible names to identify the author
        signature_words = ["sara", "shackleton", "chris", "germany", "sshackle", "cgerman"]
        for word in signature_words:
            content = content.replace(word, "")
        return content

    print("Cleaning texts...")
    combined_df['message'] = combined_df['message'].apply(clean_email)

    # 3. DATA PREPARATION FOR NAIVE BAYES
    features_train, features_test, labels_train, labels_test = train_test_split(
        combined_df['message'], combined_df['label'], test_size=0.1, random_state=42
    )

    # 4. Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)

    # 5. FEATURE SELECTION (Course only use top 10%)
    selector = SelectPercentile(f_classif, percentile=10)
    # for the decision tree excercise, i need this changed to percentile = 1(then generate new pkl files and delete old ones)
    #selector = SelectPercentile(f_classif, percentile=1)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    print("Storing processed files...")
    joblib.dump(features_train_transformed, "word_data_features_train.pkl")
    joblib.dump(features_test_transformed, "word_data_features_test.pkl")
    joblib.dump(labels_train, "emails_train_labels.pkl")
    joblib.dump(labels_test, "emails_test_labels.pkl")
    
    print("Ready to execute nb_author_id.py")

if __name__ == "__main__":
    fix_and_preprocess("emails.csv")