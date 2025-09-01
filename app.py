from flask import Flask, request, render_template
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning model

model = joblib.load('sentiment_model.pkl')

# Define the route for the main page
@app.route('/')
def home():
    """Renders the main page (index.html)."""
    return render_template('index.html')

# Define the route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    """Receives review text from the form and returns the prediction."""
    # Get the review text from the HTML form
    review_text = request.form['review_text']

    # The model expects a list of texts, so we put our text in a list
    prediction = model.predict([review_text])

    # The prediction will be an array, e.g., ['positive']. We take the first element.
    sentiment = prediction[0]

    # Render the page again, but this time with the prediction result
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment.capitalize()}')

# This block allows you to run the app directly from the command line
# We've removed the if statement to force the app to run
app.run(debug=True)