from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load model - assume it works
with open('calories.pkl', 'rb') as f:
    model_data = pickle.load(f)

if isinstance(model_data, dict):
    model = model_data['model']
    gender_encoder = model_data.get('gender_encoder', None)
    features = model_data.get('features', ['Gender_encoded', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
else:
    model = model_data
    gender_encoder = None
    features = ['Gender_encoded', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


@app.route('/cb')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    gender = request.form['gender']
    age = float(request.form['age'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    duration = float(request.form['duration'])
    heart_rate = float(request.form['heart_rate'])
    body_temp = float(request.form['body_temp'])
    
    # Encode gender
    if gender_encoder is not None:
        gender_encoded = gender_encoder.transform([gender])[0]
    else:
        gender_encoded = 1 if gender == 'male' else 0
    
    # Create input data
    input_data = pd.DataFrame([[gender_encoded, age, height, weight, 
                               duration, heart_rate, body_temp]], 
                             columns=features)
    
    predicted_calories = model.predict(input_data)[0]
    
    plt.figure(figsize=(5, 3))
    durations = [5, 10, 15, 20, 25, 30]
    calories_by_duration = []
    
    for dur in durations:
        temp_data = pd.DataFrame([[gender_encoded, age, height, weight, 
                                  dur, heart_rate, body_temp]], 
                                columns=features)
        calories_by_duration.append(model.predict(temp_data)[0])
    
    plt.plot(durations, calories_by_duration, 'b-o', linewidth=2, markersize=8, 
            label='Calories by Duration')
    plt.axhline(y=predicted_calories, color='r', linestyle='--', 
               label=f'Your Prediction: {predicted_calories:.0f} cal')
    plt.xlabel('Exercise Duration (minutes)')
    plt.ylabel('Calories Burnt')
    plt.title('Calories Burnt vs Exercise Duration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    line_chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    
    result = {
        'predicted_calories': round(predicted_calories, 1),
        'gender': gender,
        'age': age,
        'height': height,
        'weight': weight,
        'duration': duration,
        'heart_rate': heart_rate,
        'body_temp': body_temp,
        'line_chart_url': line_chart_url
    }
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)