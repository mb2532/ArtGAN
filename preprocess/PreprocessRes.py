import cv2
from matplotlib import pyplot as plt
import os 
import shutil 

# images = os.listdir("UTKFace")

# i = 0
# for f in images: 
#     val = f.split("_")
#     age = int(val[0])
#     if(age >=20 and age<60):
#       shutil.copy("UTKFace/"+f, "data/trainA")
#     i += 1
# print("Size of training set: %d" % i)

filesList = os.listdir("data/trainA")

sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "LapSRN_x2.pb"
sr.readModel(path)
sr.setModel("lapsrn", 2)
i = 0
for file in filesList: 
    img = cv2.imread("data/trainA/" + file)
    result = sr.upsample(img)
    cv2.imwrite("./data/trainA_upscaled" + file , result)
    i += 1
    if i % 100 == 0:
        print(i)
