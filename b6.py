# Use non-parametric K-Nearest Neighbor (KNN) techniques to classify grayscale images of shapes (e.g., circles, squares, and triangles). Evaluate and compare the classification accuracy of both methods.

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def generate_shape(shape,size=64):
    img=np.zeros((size,size),dtype=np.uint8)
    center=(size//2,size//2)
    if shape == 'circle':#(image, center, radius, color, thickness)
        cv2.circle(img,center,size//3,255,-1)
    if shape == 'square':#(image, left-top, right-bottom, color, thickness)
        cv2.rectangle(img, (size//4,size//4) , (size*3//4,size*3//4),255,-1)
    if shape == 'triangle':#[center-x,top] ,[left,bottom], [right,bottom]
        pts=np.array([ [size//2,size//5],  [size//5,size*4//5] ,  [size*4//5,size*4//5]], np.int32)
        cv2.fillPoly(img,[pts],255)
    return img

def create_dataset(num_samples=100):
    shapes = ['circle','square','triangle']
    x,y=[],[]
    for index,shape in enumerate(shapes):
        for n in range(num_samples):
            image = generate_shape(shape)
            x.append(image.flatten())
            y.append(index)
    return np.array(x),np.array(y)


x,y = create_dataset()
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
acc = metrics.accuracy_score(y_test,y_pred)
print(acc)

fig,axes = plt.subplots(1,5,figsize=(10,2))
for i in range(5):
    axes[i].imshow(x_test[i].reshape(64,64))
    axes[i].set_title(f"Predicted:{y_pred[i]}")
    axes[i].axis('off')
plt.show()
