#!/usr/bin/env python3
'''
Implementation of PCA on the mnist dataset. 

Jai Khanna
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df  = pd.read_csv('mnist.txt',sep=r"\s+", header=None)
    samplesPerLabel=200
    #only taking digit 0's images   
    features=df[0:samplesPerLabel*1]
    features=features.as_matrix()
    pca=PCA(features)
    k=5

    #compressing and recontructing images 0-5 in the dataframe
    for i in range(5):
        original=features[i]
        compressed=pca.compress(k,original)
        imoriginal=original
        imoriginal.shape=(16,15)
        imoriginal=np.array(imoriginal,dtype='float64')
        plt.imshow(imoriginal)
        plt.savefig('k='+str(k)+', '+str(i)+' original'+'.png')
        

        reconstruction=pca.decompress(compressed)
        imreconstruction=reconstruction
        imreconstruction.shape=(16,15)
        imreconstruction=np.array(imreconstruction,dtype='float64')
        plt.imshow(imreconstruction)
        plt.savefig('k='+str(k)+', '+str(i)+' reconstructed'+'.png')
        plt.close()

    #to get the percent of dissimilarity after reconstruction
    valuesOfk=list(range(1,240,1))
    dissimilarity=np.empty(239)
    j=0
    for i in valuesOfk:
        dissimilarity[j]=pca.getdissimilarity(i)
        j=j+1
    
    plt.plot(valuesOfk, dissimilarity, '-go',label='')
    plt.ylabel('% Dissimilarity')
    plt.xlabel('k components')
    plt.yscale('log')
    plt.legend()
    plt.show()


class PCA():
    def __init__(self, data):
        self.data=data
        self.n,self.m=np.shape(data)
        self.standardize()
    
    #subtract mean from all data points
    def standardize(self):
        mean=(self.data).mean(axis=0)
        self.centered_points=self.data-mean
        self.mean = mean
        return self.mean, self.centered_points

    #convariance
    def covariance_matrix(self):
        self.C=np.zeros((self.m,self.m))
        for i in range(self.n):
            point_transpose=self.centered_points[i].transpose()
            self.C=np.add(self.C,np.matmul(self.centered_points[i],point_transpose))
        self.C=self.C/self.n

    #SVD of covariance matrix
    def cov_SVD(self):
        self.covariance_matrix()
        self.U, self.S, self.Vt = np.linalg.svd(self.C, full_matrices=False)
        return self.U, self.S, self.Vt
    
    #projects a point to an k dim space
    def compress(self,k,point):
        self.cov_SVD()
        Uk=self.U[:,0:k]
        Ukt=Uk.transpose()
        compressed_point=np.matmul(Ukt,point)
        return compressed_point

    #projects a point back to original space 
    def decompress(self,compressed_point):
        (k,)=compressed_point.shape
        Uk=self.U[:,0:k]
        uncompressed_point=np.add(self.mean, np.matmul(Uk,compressed_point))
        return uncompressed_point
    
    #get similarity
    def getdissimilarity(self,k):
        self.cov_SVD()
        components=0
        for i in range(k+1,self.m):
            component=(self.S[i])**2
            components=components+component
        total=0
        for j in range(self.m):
            total=total+((self.S)[j])**2
        return (components/total)*100
        


if __name__ == "__main__":
    main()
