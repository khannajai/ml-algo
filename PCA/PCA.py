#!/usr/bin/env python3
'''
Implementation of PCA on the mnist dataset. 

Jai Khanna
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def main():

    data  = pd.read_csv('mnist.txt',sep=r"\s+", header=None)
    samplesPerLabel=200
    
    #only taking digit 0's images   
    data=data[0:samplesPerLabel]

    

if __name__ == "__main__":
    main()
