import  cv2
import requests

url = 'http://bcc5de97e8a8.ngrok.io/predict'

img = cv2.imread('E:/DNIM/Image/00009403/20151108_140601.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
print(type(img_encoded))
# send http request with image and receive response
response = requests.post(url, data=img_encoded.tostring())
print(response.json())