import uvicorn
from fastapi import FastAPI
import pickle
from prepare_data import CreditCard, transform

app = FastAPI()

with open("classifier.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Welcome To Credit Card Default Predictor!'}

@app.post('/predict')
def predict_default(data:CreditCard):
    data = data.dict()
    data = transform(data)
    prediction = classifier.predict(data)
    response_message = "No Default" if prediction == 0 else "Default"
    return {
        "Prediction": response_message
    }

if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8000)
    # uvicorn main:app --reload