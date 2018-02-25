import pandas as pd
import numpy as np

class LineRegression:
    
    WEIGHT_START = [70.1, 0.8]
    ETA = 1e-4
    ITERATION_COUNT = 1e4
    WEIGHT_DISTANCE_MIN = 1e-8
        
    def __init__(self, x, y):
        self.__X = x
        self.__Y = y

    #function for count w
    def countModelsWeights(self):
        transposedX = self.__X.getT()
        self.__W = np.linalg.solve(transposedX * self.__X, transposedX * self.__Y)
        print('w')
        print(self.__W)

    #fucntion for centrailizing X ....may be need seperate to other class
    def centrailizingElement(self):
        countRow =  len(self.__X)
        centrailX = self.__X.copy()
        i = 0
        
        while i < countRow:
            element = self.getElement(i, 1)
            mu = self.__countMu(i, countRow)
            sigma = self.__countSigma(mu, i, countRow)
           
            centrailElem = (element - mu) / sigma
            centrailX = centrailX.itemset((i, 1), centrailElem)
            i += 1
        
        return centrailX
    
    def __countSigma(self, mu, numberElem, countElement):   #bad name
        sumElements = 0
        i = numberElem
        while i < countElement:
            sumElements += pow((self.getElement(i, 1) - mu), 2) 
        sigma = (1 / countElement) * sumElements  
        
        return math.sqrt(sigma)
        
    def __countMu(self, numberElem, countElement):   #bad name
        sumElements = 0
        i = numberElem
        while i < countElement:
            sumElements += self.getElement(i, 1)
        mu = (1 / countElement) * sumElements
        return mu
    
    def getElement(self, row, column):
        return self.__X.item(row, column)
    
    #function for count grad_descent
    def grad_descent(self, centrailizX, y):
        n = len(centrailizX)
        i = 0
        while i < n:
            quadrature = 1/n * (y[i])
        return 1/n * quadrature

    
    #function for reverse W


testData = pd.read_csv("weights_heights.csv", sep=',')
testData.plot()
print (testData)
print (testData.loc[:, "Height"])

characteristicMatrix = np.ones((len(testData), 2))  # matrix X
for i in range(0, len(characteristicMatrix)):
    characteristicMatrix[i, 1] = testData.loc[i, "Weight"]
characteristicMatrix = np.matrix(characteristicMatrix)
print ('characteristicMatrix')
print (characteristicMatrix)


heightMatrix = np.ones((len(testData), 1))  # matrix X
for i in range(0, len(heightMatrix)):
    heightMatrix[i, 0] = testData.loc[i, "Height"]

print ('heightMatrix')
print (heightMatrix)

lineRegression = LineRegression(characteristicMatrix, heightMatrix)

# 2 number develop - REMOVE THIS COMMENT
lineRegression.countModelsWeights()
centrailizX = lineRegression.centrailizingElement()
# w 
lineRegression.grad_descent(centrailizX, heightMatrix)

