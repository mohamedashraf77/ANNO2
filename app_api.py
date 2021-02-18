# app.py
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
#from flask_ngrok import run_with_ngrok
from predict import predict
import cv2
import numpy as np

APP = Flask(__name__)
#run_with_ngrok(APP)
API = Api(APP)


class Predict(Resource):

    @staticmethod
    def post():
        r = request
        # convert string of image data to uint8
        nparr = np.fromstring(r.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(type(img))
        prediction = predict(img)
        response = 'day'
        if prediction[0]>0.5:
          response = 'night'
        out = {'Prediction': response}

        return out, 200


API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run()