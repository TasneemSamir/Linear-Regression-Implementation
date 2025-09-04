import random
import matplotlib.pyplot as plt 
import numpy as np

#=========Tricks=========
def simpleTrick(bais,weight,x,y,eta1=0.005,eta2=0.005):
    # predicted value 
    y_predicted= weight * x + bais
    #actual value is above the predected value 
    if y_predicted < y:
        #case 1:and right y-axis
        if x >= 0:

            weight=weight+eta1
            bais=bais+eta2
        #case2: and left y-axis
        else:
            weight = weight - eta1
            bais = bais + eta2
    #actual value is below the predicted value 
    else:
    #case3: and right y-axis
        if x >= 0:
            weight = weight - eta1
            bais= bais-eta2
        else:
            weight = weight + eta1
            bais= bais - eta2
    return bais, weight

def squareTrick(bais,weight,x,y,eta1=0.005,eta2=0.005):
    #make prediction 
    y_predicted = weight * x + bais
    #rotate the line 
    weight = weight + eta1 * x * ( y - y_predicted)
    #translate the line 
    bais= bais + eta2 * (y - y_predicted)
    return bais, weight

def absoluteTrick (bais, weight,x,y,eta1=0.005,eta2=0.005):
    y_predicted=weight*x+bais

    if y > y_predicted:

        weight=weight+eta1*x
        bais=bais+eta2

    else:

        weight=weight-eta1*x
        bais=bais-eta2

    return bais, weight

#========linear regression algorithm=========
def linearRegression(features,label,epochs=50,eta1=0.005,eta2=0.005,method='square'):
    weight = random.random()
    bais = random.random()

    for epoch in range(epochs):

        i= random.randint(0, len(features)-1)
        x= features[i]
        y=label[i]

        if method =='square':
            bais,weight=squareTrick(bais,weight,x,y,eta1=eta1,eta2=eta2)
        elif method =='simple':
            bais,weight=simpleTrick(bais,weight,x,y,eta1=eta1,eta2=eta2)
        elif method =='absolute':
            bais,weight=absoluteTrick(bais,weight,x,y,eta1=eta1,eta2=eta2)
    return bais,weight

#==========Loss Functions==========
def square_loss(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) **2)

def absolute_loss(y_actual, y_predicted):
    return np.mean(np.abs(y_actual - y_predicted))

def rmse(y_actual,y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted) **2))

#================Gradient Descent================
def gradientDescent(w, b, X, y, learning_rate=0.001, epochs=1000):
    n = len(X)

    for epoch in range(epochs) :
        #partial derv. to mse withrespect to w
        dw = -(2/n) * np.sum (X * (y - (w * X + b)))
        #partial derv. to mse withrespect to b
        db= -(2/n) * np.sum (y - (w * X + b))

        w -= dw * learning_rate
        b -= db * learning_rate

    return w, b

#=========main===============
features = np.array([1, 2, 3, 4, 5, 6, 7])
labels  =np.array([155, 197, 244, 300, 356, 407, 448])

b_simple,w_simple=linearRegression(features, labels, epochs=1000, method='simple')
b_square,w_square=linearRegression(features, labels, epochs=1000, method='square')
b_abs,w_abs=linearRegression(features, labels, epochs=1000, method='absolute')

#predictions
y_pred_simple = w_simple * features + b_simple
y_pred_square = w_square * features + b_square
y_pred_abs    = w_abs * features + b_abs

print("Simple Trick: y =", w_simple, "x +", b_simple, " | RMSE =", rmse(labels, y_pred_simple))
print("Square Trick: y =", w_square, "x +", b_square, " | RMSE =", rmse(labels, y_pred_square))
print("Absolute Trick: y =", w_abs, "x +", b_abs, " | RMSE =",rmse(labels, y_pred_abs))

#test with gradient 
final_w, final_b = gradientDescent(w_square, b_square, features, labels, learning_rate=0.01, epochs=1100)
y_pred = final_w * features + final_b
print("Final Model: y =", final_w, "X +", final_b)
print("RMSE (square trick with gradient descent)=", rmse(labels, y_pred))

#=========Visualization======
plt.figure(figsize=(15,5))

# square trick
plt.subplot(1,4,1)
plt.scatter(features, labels, color='blue')
plt.plot(features, [w_square*x+b_square for x in features], color='red')
plt.title("Square Trick")

# simple trick
plt.subplot(1,4,2)
plt.scatter(features, labels, color='blue')
plt.plot(features, [w_simple * x + b_simple for x in features], color = 'green')
plt.title("Simple Trick")

# absolute trick
plt.subplot(1,4,3)
plt.scatter(features, labels, color ='blue')
plt.plot(features, [ w_abs * x+ b_abs for x in features], color='purple')
plt.title("Absolute Trick")

#square with gradient descent 
plt.subplot(1,4,4)
plt.scatter(features, labels, color ='blue')
plt.plot(features, [ final_w * x+ final_b for x in features], color='yellow')
plt.title("Square Trick with gradient")

plt.show()
