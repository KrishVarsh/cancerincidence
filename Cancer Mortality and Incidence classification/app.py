
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('naivemodel.pkl')

# Mapping dictionary for status descriptions
status_mapping = {
    1: 'Falling',
    2: 'Stable',
    3: 'Rising'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        index = float(request.form['index'])
        fips = float(request.form['fips'])
        met_objective = float(request.form['met_objective'])
        death_rate = float(request.form['death_rate'])
        lower_ci = float(request.form['lower_ci'])
        upper_ci = float(request.form['upper_ci'])
        avg_deaths = float(request.form['avg_deaths'])
        recent_trend = float(request.form['recent_trend'])
        recent_5year_trend = float(request.form['recent_5year_trend'])
        trend_lower_ci = float(request.form['trend_lower_ci'])
        trend_upper_ci = float(request.form['trend_upper_ci'])
        
        # Perform prediction based on the form data
        prediction = model.predict([[index, fips, met_objective, death_rate, lower_ci, upper_ci, avg_deaths, recent_trend, recent_5year_trend, trend_lower_ci, trend_upper_ci]])

        # Map the prediction to status description
        status = status_mapping.get(prediction[0], 'Unknown')

        # Pass the prediction and status to the template
        return render_template('predict.html', prediction=prediction[0], status=status)
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
