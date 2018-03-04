import pandas as pd
import numpy as np
import math

class LineRegression:
    
    ETA = 1e-4
    ITERATION_COUNT = 1e4
    WEIGHT_DISTANCE_MIN = 1e-8
        
    def __init__(self, x, y):
        self.__X = x
        self.__Y = y

    #function for count w
    def countModelsWeights(self):
        transposedX = self.__X.getT()
        return np.linalg.solve(transposedX * self.__X, transposedX * self.__Y)

    #fucntion for centrailizing X ....may be need seperate to other class
    def centrailizingElement(self):
        countRow =  len(self.__X)
        centrailX = self.__X.copy()
        i = 0
        
        while i < countRow:
            element = self.__X.item(i, 1)
            mu = self.__countMu(self.__X, i, countRow)
            sigma = self.__countSigma(self.__X, mu, i, countRow)
           
            centrailElem = (element - mu) / sigma
            centrailX = centrailX.itemset((i, 1), centrailElem)
            i += 1
        
        return centrailX
    
    # I don't know how implement it. google also :(
    def reverseCentrailizingElement(self, elem):
        return elem
    
    def __countSigma(self, elem, mu, numberElem, countElement):   #bad name
        sumElements = 0
        i = numberElem
        while i < countElement:
            sumElements += pow((elem.item(i, 1) - mu), 2) 
        sigma = (1 / countElement) * sumElements  
        
        return math.sqrt(sigma)
        
    def __countMu(self, elem, numberElem, countElement):   #bad name
        sumElements = 0
        i = numberElem
        while i < countElement:
            sumElements += elem.item(i, 1)
        mu = (1 / countElement) * sumElements
        return mu
    
    #function for count grad_descent
    def grad_descent(self, centrailizX, y, wStart):
        allElemInCentrailX = len(centrailizX)
        wOld = np.array(wStart)
        wNew = np.array([])
        
        i = 0
        condition = True
        while condition:
            wNew = self.countWGradientDescent(centrailizX, wOld, y, allElemInCentrailX)
            if np.norm(wNew - wOld) < self.WEIGHT_DISTANCE_MIN or i >= self.ITERATION_COUNT:
                condition = False
            wOld = wNew
            i += 1
        return wNew
    
    #function for count W on one step
    def countWGradientDescent(self, centrailizX, w, y, allElemInCentrailX):
        first = w[0] - self.ETA * self.countFirst(centrailizX, w, y, allElemInCentrailX)
        second = w[1] - self.ETA * self.countSecond(centrailizX, w, y, allElemInCentrailX)
        return np.array([first, second])
    
    def countFirst(self, centrailizX, w, y, allElemInCentrailX):
        gradFirst = 0
        i = 0
        while i < allElemInCentrailX:
            gradFirst += self.countElemForGradFirst(centrailizX, w, y, i)
        return 2/allElemInCentrailX * gradFirst
    
    def countElemForGradFirst(self, centrailizX, w, y, numberElement):
        return (w[0] + centrailizX.item(numberElement, 0) * w[1]) - y.loc[numberElement, "Height"]
    
    def countSecond(self, centrailizX, w, y, allElemInCentrailX):
        gradSecond = 0
        i = 0
        while i < allElemInCentrailX:
            gradSecond += self.countElemForGradSecond(centrailizX, w, y, i)
        return 2/allElemInCentrailX * gradSecond
    
    def countElemForGradSecond(self, centrailizX, w, y, numberElement):
        return ((w[1] + centrailizX.item(numberElement, 0) * w[1]) - y.loc[numberElement, "Height"]) * centrailizX.item(numberElement, 0)
    
    def countOnRandomElem(self, centrailizX, y, w, numberElement):
        first = w[0] - self.ETA * self.countElemForGradFirst(centrailizX, w, y, numberElement)
        second = w[1] - self.ETA * self.countElemForGradSecond(centrailizX, w, y, numberElement)
        return np.array([first, second])
    
    

testData = pd.read_csv("weights_heights.csv", sep=',')
testData.plot()
#print (testData)
#print (testData.loc[:, "Height"])

characteristicMatrix = np.ones((len(testData), 2))  # matrix X
for i in range(0, len(characteristicMatrix)):
    characteristicMatrix[i, 1] = testData.loc[i, "Weight"]
characteristicMatrix = np.matrix(characteristicMatrix)
#print ('characteristicMatrix')
#print (characteristicMatrix)


heightMatrix = np.ones((len(testData), 1))  # matrix X
for i in range(0, len(heightMatrix)):
    heightMatrix[i, 0] = testData.loc[i, "Height"]

#print ('heightMatrix')
#print (heightMatrix)

lineRegression = LineRegression(characteristicMatrix, heightMatrix)

# 2 number develop - REMOVE THIS COMMENT

# 1 answer
wAnalitic = lineRegression.countModelsWeights()
print('wAnalitic')
print(wAnalitic)

#2 answer
centrailizX = lineRegression.centrailizingElement()

wGrad = lineRegression.grad_descent(centrailizX, heightMatrix, [70.1, 0.8])
lineRegression.reverseCentrailizingElementwGrad(wGrad) # not work
print ('wGrad')
print (wGrad)

# 3 answer
np.random.seed(42)
numberElement = np.random.randint(centrailizX.shape[0])

wGradOnOneELem = lineRegression.countOnRandomElem(centrailizX, heightMatrix, [70.1, 0.8], numberElement)
print ('wGradOnOneELem')
print (wGradOnOneELem)