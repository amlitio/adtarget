# adtarget

Personalized Ad Recommendation System  (twirl~~ still in template form)
This repository contains a personalized ad recommendation system built using Python, Flask, scikit-learn, and scikit-surprise. The system employs collaborative filtering, content-based filtering, and a RandomForest classifier to provide users with personalized ad recommendations based on their age, daily internet usage, time spent on the site, and gender.

-Features
Flask-based web application for user input and displaying recommendations
Collaborative filtering using the SVD algorithm from the scikit-surprise library
Content-based filtering using the TfidfVectorizer and cosine_similarity from the scikit-learn library
RandomForest classifier for predicting user clicks

-Installation:
1 Clone the repository:
Copy code

git clone https://github.com/your_username/personalized_ad_recommendation.git


2 Change to the cloned directory:
Copy code

cd personalized_ad_recommendation


3 Install the required Python packages:
Copy code

pip install -r requirements.txt



-Usage

1 Update app.py with your dataset and preprocessing steps.
2 Run the Flask application:

Copy code
python app.py


3 Open your web browser and go to http://localhost:5000/.
4 Enter your information (age, daily internet usage, time spent on site, and gender) and click "Get Recommendations" to receive personalized ad recommendations.


-Customization

You can customize the ad recommendation system by:

'Updating the dataset and preprocessing steps in app.py.
'Modifying the collaborative filtering and content-based filtering algorithms in app.py.
'Adjusting the RandomForest classifier parameters in app.py.


License
This project is licensed under the MIT License. See the LICENSE file for details.
