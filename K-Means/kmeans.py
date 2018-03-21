#!/usr/bin/env python3
'''
Implementation of k-means clustering on the mnist dataset. 

Jai Khanna
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance

def main():

    data  = pd.read_csv('mnist.txt',sep=r"\s+", header=None)
    samplesPerLabel=200
    
    #only taking digit 0's images   
    data=data[0:samplesPerLabel]

    #performing k-means clustering
    k=1
    codebook, _= calcKmeans(data, k)
    (nrows,_)=np.shape(codebook)

    #showing images of codebook vectors
    for i in range(nrows):
        image=codebook[i]
        image.shape=(16,15)
        image=np.array(image,dtype='float64')
        plt.imshow(image)
        plt.savefig('k='+str(k)+', '+str(i)+'.png')
        plt.close()

def calcKmeans(data,k):
    data=data.as_matrix()
    (nrows,ncol)=np.shape(data)

    #randomly assigning labels
    labels=np.random.randint(k, size=nrows)

    codebook=np.zeros((k,ncol))
    labelsNew=np.zeros(nrows)

    while (True):
        #calculating codebook vectors
        for x in range(k):
            sum=np.zeros(ncol)
            j=0
            for i in range(nrows):
                if labels[i]==x:
                    j=j+1
                    sum=np.add(sum,data[i])
            mean=np.true_divide(sum,j)
            codebook[x]=mean

        #putting datapoints in closest clusters
        for x in range(nrows):
            mini=10000
            minIndex=0
            for i in range(k):
                dist = np.linalg.norm(data[x]-codebook[i])
                if dist<mini:
                    mini=dist
                    minIndex=i
            labelsNew[x]=minIndex
        
        #break the loop is arrays are equal
        if(np.array_equal(labelsNew,labels)):
            break
        
        labels=labelsNew
       
    return codebook, labelsNew

if __name__ == "__main__":
    main()
