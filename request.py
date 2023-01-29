import requests
from PIL import Image


# img = Image.open(imgFile)
# print (img)
for num in range(5):
	imgFile = open("/Users/abhinavkbij/Downloads/digit.jpeg", "rb")
	r = requests.post("http://20.244.8.129:8000/identifyDigit", files={"imgFile":("imgFile", imgFile)})
	print (r.text)