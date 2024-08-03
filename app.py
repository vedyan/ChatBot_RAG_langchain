''' This code snippet creates a Flask web application for predicting forest fire
occurrences using the trained logistic regression model (model.pkl).'''

from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Loads the trained logistic regression model (model.pkl) using pickle.
# load() and assigns it to the variable model.
model=pickle.load(open('model.pkl','rb'))

# Initializes a Flask application named app.
# Defines a route '/' for the home page of the web application. When users visit this route,
# it renders the HTML template named "forest.html".
@app.route('/')
def hello_world():
    return render_template("forest.html") # CONNECT WITH FRONTEND

# Defines a route '/predict' which handles both GET and POST requests.
# This route is responsible for predicting forest fire occurrences based
# on the input features provided by the user.
@app.route('/predict',methods=['POST','GET']) #DEFINES ENDPOINT FOR MAKING PREDICTIONS
# HTTP Methods : GET, POST, PUT, DELETE, etc / POST: to send data to the server for processing.
def predict():
#  extracts the input features submitted by the user through a form
#  (using request.form.values()), converts them to integers,
#  and stores them in the list int_features.
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)] #  NumPy array final containing the input features.
    print(int_features)
    print(final)
# Uses the trained logistic regression model (model) to predict the probability of forest
# fire occurrence using the predict_proba() method and stores it in the variable prediction.
    prediction=model.predict_proba(final)
# Formats the prediction probability to display only two decimal places.
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('forest.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('forest.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
'''If the predicted probability is greater than 0.5, it indicates a higher likelihood of 
forest fire occurrence, and a warning message is displayed. Otherwise, it displays 
a message indicating that the forest is safe.'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
