from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

app = Flask(__name__)

# Load and preprocess your dataset
# ...
data = ...

# Collaborative filtering using SVD
# ...
svd = ...

# Content-based filtering using TfidfVectorizer and cosine_similarity
# ...
tfidf_matrix = ...
cosine_sim = ...

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3, random_state=42)

# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Define function for predicting clicks
def predict_clicks(age, daily_internet_usage, time_spent_on_site, male):
    user = [[age, daily_internet_usage, time_spent_on_site, male]]
    return rf.predict(user)[0]

# Define function for recommending ads
def recommend_ad(age, daily_internet_usage, time_spent_on_site, male):
    ad_recommendations = []
    
    # Use your collaborative filtering and content-based filtering results here to generate personalized ad recommendations
    # ...
    
    return ad_recommendations[:3]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = int(request.form["age"])
        daily_internet_usage = float(request.form["daily_internet_usage"])
        time_spent_on_site = float(request.form["time_spent_on_site"])
        male = int(request.form["gender"])

        recommended_ads = recommend_ad(age, daily_internet_usage, time_spent_on_site, male)
        return render_template("results.html", recommended_ads=recommended_ads)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
