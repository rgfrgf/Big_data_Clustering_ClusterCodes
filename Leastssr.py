# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%



# %%
import numpy as np
import pandas as pd
from scipy.linalg import hankel
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data=pd.read_csv('Comp1_IE529.csv',header=None)



X=data[0].values
Y=data[1].values
avgy=np.mean(Y)
SStotal=sum(np.power(Y-avgy,2))


q=4 #order of the polynomial regression
#input paraemeter

H1=[]
H2=[]
Ybar=[]
i=0
for i in range(0,q+1):
    x=sum(np.power(X,i))
    y=sum(np.power(X,i)*Y)
    H1.append(x)
    Ybar.append(y)
    i+=1
for i in range(q,2*q+1):
    x=sum(np.power(X,i))
    H2.append(x)#H2 is the last row of the hankel matrix
    i+=1
H=hankel(H1,H2)
#use hankel matrix to calculate optimized beta: beta=inv(hankel)*Ybar

beta=np.matmul(inv(H),Ybar)
beta


mid=[]
ypredict=[]
for j in range(0,len(X)):
    mid=[]
    for i in range(0,len(beta)):
        yp=(np.power(X[j],i))*beta[i]
        mid.append(yp)
        y=sum(mid)
    ypredict.append(y)

SSres=sum(np.power(Y-ypredict,2))
Rsquare=1-SSres/SStotal
print(SSres,Rsquare)

plt.scatter(X,Y)
plt.xlabel('lift(kg)')
plt.ylabel('putt(m)')
results=plt.plot(np.sort(X),np.sort(ypredict))
plt.setp(results, color='r', linewidth=2.0)

regr=LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter=100000,C=1000.0,tol=0.01)


Y_int=Y.astype(int)
XX=[X]
X

XXX=np.array(XX).reshape(-1,1)
YY=[Y_int]
YYY=np.array(YY).reshape(-1,1)
YYY

regr.fit(XXX,YYY)

Y_pred=regr.predict(XXX)
Y_pred
idx=np.argsort(X)
idx

plt.scatter(X,Y)
plt.xlabel('lift(kg)')
plt.ylabel('putt(m)')
results=plt.plot(XXX[idx],(Y_pred[idx]))
plt.setp(results, color='r', linewidth=1.0)
XXX[idx]


SSres=sum(np.power(Y-Y_pred,2))
Rsquare=1-SSres/SStotal
print(SSres,Rsquare)

