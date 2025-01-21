import streamlit as st
import pandas as pd
#import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud

# Download NLTK resources if not already downloaded
nltk.download('stopwords')

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to preprocess the reviews
def preprocess_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)  # Remove non-alphabetic characters
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Split into words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  # Remove stopwords and stem
    return ' '.join(review)  # Return the processed review

# Streamlit App
st.title("Customer Review Sentiment Analysis")

    
# File uploader for the dataset
dataset = st.file_uploader(r"Restaurant_Reviews.tsv",type= ["tsv"])
#data = pd.read_csv(r'Restaurant Review.py')
if dataset is not None:
    # Load dataset
    dataset = pd.read_csv(dataset, delimiter='\t', quoting=3)
    st.write("Dataset Loaded Successfully!")
    st.dataframe(dataset.head())
    # Preprocess reviews
    
    
    corpus = []
    for i in range(len(dataset)):
        processed_review = preprocess_review(dataset['Review'][i])
        corpus.append(processed_review)

    # Vectorize the reviews
    cv = CountVectorizer()
    X = cv.fit_transform(corpus).toarray()
    y = dataset['Liked'].values  # Assuming the sentiment column is named 'Sentiment'

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Train the Logistic Regression model
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    
    # Input for new review prediction
    new_review = st.chat_input("Enter a new review for sentiment prediction:")
    
    if st.button("Predict Sentiment"):
        if new_review:
            # Preprocess the new review
            processed_new_review = preprocess_review(new_review)
            new_review_vectorized = cv.transform([processed_new_review]).toarray()
            prediction = classifier.predict(new_review_vectorized)

            # Display the prediction
            res = st.write("Predicted Sentiment:", "Positive" if prediction[0] ==1 else "Negative")
            if  prediction[0] == 1:
                st.image("like.jpg", caption="Good Review",width=50)
            elif prediction[0] == 0:
                st.image("dislike.jpg", caption="Bad Review",width=50)
                st.subheader("Sentiment Distribution")
                st.subheader("Word Cloud for Negative Feedback")
                negative_reviews = " ".join([corpus[i] for i in range(len(corpus)) if y[i] == 1])
                wordcloud_neg = WordCloud(width=400, height=200, background_color="white").generate(negative_reviews)
                st.image(wordcloud_neg.to_array(), caption="Negative Feedback", use_column_width=True)

# Display sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = pd.Series(y_train).value_counts()
    st.bar_chart(sentiment_counts,color="#ffaa0088",horizontal=True)

    # Generate word clouds for positive and negative feedback
    st.subheader("Word Cloud for Positive and Negative Feedback")
    positive_reviews = " ".join([corpus[i] for i in range(len(corpus)) if y[i] == 1])
    negative_reviews = " ".join([corpus[i] for i in range(len(corpus)) if y[i] == 0])
    from wordcloud import WordCloud
    # Word clouds
    wordcloud_pos = WordCloud(width=400, height=200, background_color="white").generate(positive_reviews)
    wordcloud_neg = WordCloud(width=400, height=200, background_color="white").generate(negative_reviews)
    
    st.image(wordcloud_pos.to_array(), caption="Positive Feedback", use_column_width=True)
    st.image(wordcloud_neg.to_array(), caption="Negative Feedback", use_column_width=True)
    

#streamlit run rr1_app.py
