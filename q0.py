# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler



#reading data
data=pd.read_csv('3DPCAdata.csv',header=None)
data=np.array(data)



#SVD OF Origin Matrix
u, s, vh = np.linalg.svd(data, full_matrices=False)
u



# debias the matrix
mean=data.mean(axis=1)
n=data.shape[1]
mean=np.outer(mean, np.ones(n))
debias = data-mean
debias.shape



debiast=np.transpose(debias)
covariance=np.matmul(debias,debiast)
#normalize
covariance=covariance/(debias.shape[1]-1)
u, s, vh = np.linalg.svd(debias,full_matrices=False)
vv=np.transpose(vh)
#covariance



value,vector=np.linalg.eig(covariance)
idx = value.argsort()[::-1]   
eigenValues = value[idx]
eigenVectors = vector[:,idx]
pc=eigenVectors[:,0]
eigenVectors[:,1]



#data visualization with eigenvectors plotted
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs=debias[0]
ys=debias[1]
z=debias[2]
ax.scatter(xs, ys, z, zdir='z', s=20, c=None, depthshade=True)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
xs2=pc[0]
ys2=pc[1]
z2=pc[2]
ax.quiver(0,0,0,eigenVectors[:,0][0]*eigenValues[0]*199,eigenVectors[:,0][1]*eigenValues[0]*199,eigenVectors[:,0][2]*eigenValues[0]*199,length=0.1, normalize=False)
ax.quiver(0,0,0,eigenVectors[:,1][0]*eigenValues[1]*199,eigenVectors[:,1][1]*eigenValues[1]*199,eigenVectors[:,1][2]*eigenValues[1]*199,length=0.1, normalize=False)
ax.quiver(0,0,0,eigenVectors[:,2][0]*eigenValues[2]*199,eigenVectors[:,2][1]*eigenValues[2]*199,eigenVectors[:,2][2]*eigenValues[2]*199,length=0.1, normalize=False)



debias.shape



project=np.matmul(np.transpose(debias),eigenVectors[:,0:2])
project



plt.scatter(project[:,0],project[:,1])



standard=StandardScaler().fit_transform(data)



uu, ss, vvh = np.linalg.svd(debias, full_matrices=False)

