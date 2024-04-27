from django.shortcuts import render
from django.http import JsonResponse
from joblib import load
import numpy as np

# Create your views here.
def index(request):
    return render(request, 'myapp/index.html')

def predict(request):
    if request.method == 'POST':
        # Load your machine learning model
        model = load('myapp/model/crop_yield_linear_regression_model.joblib')
        scaler = load('myapp/model/standard_scaler.joblib') 

        # Get the input values from the form and convert them to float
        rainfall = float(request.POST.get('rainfall'))
        fertilizer = float(request.POST.get('fertilizer'))
        temperature = float(request.POST.get('temperature'))
        nitrogen = float(request.POST.get('nitrogen'))
        phosphorus = float(request.POST.get('phosphorus'))
        potassium = float(request.POST.get('potassium'))

        # Print the received input data for debugging
        print("Received input data:")
        print(f"Rainfall: {rainfall}")
        print(f"Fertilizer: {fertilizer}")
        print(f"Temperature: {temperature}")
        print(f"Nitrogen: {nitrogen}")
        print(f"Phosphorus: {phosphorus}")
        print(f"Potassium: {potassium}")

        # Prepare input data array and scale it
        input_data = np.array([[rainfall, fertilizer, temperature, nitrogen, phosphorus, potassium]])
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Print the prediction for debugging
        print(f"Predicted crop yield: {prediction[0]}")

        # Convert the prediction ndarray to a Python list
        prediction_list = prediction.tolist()

        # Return the prediction as a JSON response
        return JsonResponse({'prediction': prediction_list})

    return render(request, 'myapp/predict.html')
