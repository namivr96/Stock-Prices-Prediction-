import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Function for processing training and testing data
def getdata(data,ln):
    X,Y = [],[]
    for i in range( len(data)-ln-1 ):
        X.append(data[ i:(i+ln),0] )
        Y.append(data[ (i+ln),0] )
    return np.array(X),np.array(Y)
	
#Function for loading input data
data = pd.read_csv('C:/Users/namra/Desktop/Fall 2018/ECE 629/project/nasdaq100_padding.csv')
dt = data['ADBE']
dt.dropna(inplace=True)

#Function for scaling input data 
scale = MinMaxScaler()
dt = dt.values.reshape(dt.shape[0],1)
dt = scale.fit_transform(dt)
dt

#Spliting input data into training and testing data
X,y = getdata(dt,7)
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]

#Parameters of the model 
model = Sequential()
model.add(LSTM(256,input_shape=(7,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

#Training the model 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
history = model.fit(X_train,y_train,epochs = 20,validation_data=(X_test,y_test),shuffle=False)

#Testing the model using test data 
Xt = model.predict(X_test)

#Plotting actual data vs predicted data
plt.rcParams.update({'font.size': 18}) 
plt.plot(scale.inverse_transform(y_test.reshape(-1,1)), color='red', label='Actual value')
plt.plot(scale.inverse_transform(Xt), color='green', label='Predicted Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.xlabel('Number of days')
plt.ylabel('Closing Prices')
plt.title('Closing prices prediction for ADBE (Small dataset)')
plt.show()

#Calculating the mean squared error
xin = scale.inverse_transform(Xt)
yin = scale.inverse_transform(y_test.reshape(-1,1))
final_mse  = mean_squared_error(xin, yin)
print ("Mean squared error after predictions: %f"%(final_mse))

#Tracking the trends between actual and predicted data 
Xt_int = scale.inverse_transform(Xt.reshape(-1,1))
yt_int = scale.inverse_transform(y_test.reshape(-1,1))
xt_class = [Xt_int[i+1]-Xt_int[i]for i in range (0,len(Xt_int)-1)]
yt_class = [yt_int[i+1]-yt_int[i] for i in range (0,len(yt_int)-1)]

cnt = 0 
for i in range (len(yt_class)):
  if yt_class[i] <= 0 and xt_class[i] <= 0:
    cnt = cnt + 1
  if yt_class[i] > 0 and xt_class[i] > 0:
    cnt = cnt + 1


##MSE
xin = scale.inverse_transform(Xt)
yin = scale.inverse_transform(y_test.reshape(-1,1))
final_mse  = mean_squared_error(xin, yin)
print ("Mean squared error after predictions: %f"%(final_mse))
