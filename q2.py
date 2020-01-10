# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import numpy as np
import pandas as pd
from scipy.linalg import hankel
from sklearn.decomposition import PCA
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


data=pd.read_csv('PCA_comp1.csv',header=None)

mean=pd.array(data.mean(axis=0))
np.transpose(mean)
meantominus=pd.DataFrame(np.outer(np.ones((data.shape[0],)),mean))
debiased=data-meantominus
debiased=np.array(debiased)

covariance=np.matmul(debiased,np.transpose(debiased))/(debiased.shape[0]-1)
#pd.DataFrame(covariance)


eigenvalue,eigenvector=np.linalg.eig(covariance)
idx=np.argsort(eigenvalue)[::-1]
eigenvectorsort=eigenvector[idx]
eigenvector[1,:]


pca1=PCA(n_components=4)


standard=StandardScaler().fit_transform(data)
data_lowdimension=pca1.fit(standard).transform(standard)
ax=plt.figure()
for c,i in zip('rgb',[0,1]):
   plt.scatter(data_lowdimension[:,0],data_lowdimension[:,1],c=c)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')


# pca result evaulation
print('pca1.components',pca1.components_)

print('pca1.explained_variance_ratio_',pca1.explained_variance_ratio_)

print('pca1.explained_variance_',pca1.explained_variance_)


pca2=PCA(n_components=2)#debiased bata input
data_lowdimension=pca2.fit(debiased).transform(debiased)
ax=plt.figure()
for c,i in zip('rgb',[0,1]):
   plt.scatter(data_lowdimension[:,0],data_lowdimension[:,1],c=c)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')

# pca result evaulation
print('pca2.explained_variance_ratio_',pca2.explained_variance_ratio_)


# %%
print('pca2.explained_variance_',pca2.explained_variance_)

C=np.array([2, 3, 8, 1, 5])
A=np.array([3, 5, 11, 2, 7])
np.argsort(C/A)


n=98
sum=0
while(n>0):
    sum+=(n%10)*(n%10)
    n/=10



