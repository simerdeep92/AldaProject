#install.packages("e1071")
#install.packages("klaR")
library(caret)
library("e1071")
library("klaR")
#csvData <- file.choose()
csvData <- read.csv("output_tfidf.csv", header=T, sep=',')

#labelData <- file.choose()
labelData <- read.csv("Discussion_Category_Less5_2.csv", header=T, sep=',')

labelList <- as.list(as.data.frame(t(labelData)))
integData <- cbind(csvData, labelData)
trainingDataSize <- floor(0.70 * nrow(integData))
set.seed(123)
train_ind <- sample(seq_len(nrow(integData)), size = trainingDataSize)
trainingData <- integData[train_ind, ]
testData <- integData[-train_ind, ]

numOfCols <- ncol(csvData)
xtrain = trainingData[,1:numOfCols]
ytrain = trainingData[,numOfCols+1]
ytrain <- as.factor(ytrain)

###########PCA#######
#xtrainT <- t(xtrain)
#p <- prcomp(xtrain, retx=TRUE, center=TRUE, scale=TRUE)
p <- prcomp(xtrain)
#print(summary(p))
plot(p, type = "l")

# Reducing dimensions to 6 after experimenting in PCA_Naive_Test.R
i <- 6
pRot <- (p$rotation[,1:i])

xtrainMat <- data.matrix(xtrain)
xtrainMult <- xtrainMat %*% pRot

xtest = testData[,1:numOfCols]
ytest = testData[,numOfCols+1]

##########PCA##########
#xtestT <- t(xtest)
pTest <- prcomp(xtest)
#print(summary(p2))
pRotTest <- (pTest$rotation[,1:i])

xtestMat <- data.matrix(xtest)
xtestMult <- xtestMat %*% pRotTest

xtrainMultData <- as.data.frame(xtrainMult)

precision = c()
recall=c()
FMeasure=c()
accuracy = c()
classAccuracy = c()
classPrecision = c()
classRecall = c()
classFmeasure = c()
avgAccuracy = c()
avgRecall = c()
avgPrecision = c()
avgFmeasure = c()
method = c("logistic regression");
#tuneLinear = best.tune(svm,train.x=xtrain, train.y=ytrain,kernel ="linear")
#print(summary(tuneLinear))
#linearModel <- naiveBayes(ytrain ~., data=xtrainMultData)
xtrainMultDataFrame <- as.data.frame(xtrainMultData)
linearModel <- glm(ytrain ~., family=binomial(link='logit'), data=xtrainMultDataFrame)

xtestMultDataFrame <- as.data.frame(xtestMult)

predQuadratic <- predict(linearModel, xtestMultDataFrame)
confusionMatrix = matrix(nrow = 5, ncol = 5)
confusionMatrix <- table(predQuadratic,ytest)
A1 <- confusionMatrix[1,1]
A2 <- confusionMatrix[1,2]
A3 <- confusionMatrix[1,3]
A4 <- confusionMatrix[1,4]
A5 <- confusionMatrix[1,5]
B1 <- confusionMatrix[2,1]
B2 <- confusionMatrix[2,2]
B3 <- confusionMatrix[2,3]
B4 <- confusionMatrix[2,4]
B5 <- confusionMatrix[2,5]
C1 <- confusionMatrix[3,1]
C2 <- confusionMatrix[3,2]
C3 <- confusionMatrix[3,3]
C4 <- confusionMatrix[3,4]
C5 <- confusionMatrix[3,5]
D1 <- confusionMatrix[4,1]
D2 <- confusionMatrix[4,2]
D3 <- confusionMatrix[4,3]
D4 <- confusionMatrix[4,4]
D5 <- confusionMatrix[4,5]
E1 <- confusionMatrix[5,1]
E2 <- confusionMatrix[5,2]
E3 <- confusionMatrix[5,3]
E4 <- confusionMatrix[5,4]
E5 <- confusionMatrix[5,5]

sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy = (A1+B2+C3+D4+E5)/sumOfElements
resultTable = data.frame(method,accuracy)
print(resultTable)
classAccuracy[1] = A1/sumOfElements
classAccuracy[2] = B2/sumOfElements
classAccuracy[3] = C3/sumOfElements
classAccuracy[4] = D4/sumOfElements
classAccuracy[5] = E5/sumOfElements

classPrecision[1] = A1/(A1+B1+C1+D1+E1)
classPrecision[2] = B2/(A2+B2+C2+D2+E2)
classPrecision[3] = C3/(A3+B3+C3+D3+E3)
classPrecision[4] = D4/(A4+B4+C4+D4+E4)
classPrecision[5] = E5/(A5+B5+C5+D5+E5)

classRecall[1] = A1/(A1+A2+A3+A4+A5)
classRecall[2] = B2/(B1+B2+B3+B4+B5)
classRecall[3] = C3/(C1+C2+C3+C4+C5)
classRecall[4] = D4/(D1+D2+D3+D4+D5)
classRecall[5] = E5/(E1+E2+E3+E4+E5)

classFmeasure[1] <- (2 * ( classPrecision[1] * classRecall[1]) / ( classPrecision[1] + classRecall[1] ))
classFmeasure[2] <- (2 * ( classPrecision[2] * classRecall[2]) / ( classPrecision[2] + classRecall[2] ))
classFmeasure[3] <- (2 * ( classPrecision[3] * classRecall[3]) / ( classPrecision[3] + classRecall[3] ))
classFmeasure[4] <- (2 * ( classPrecision[4] * classRecall[4]) / ( classPrecision[4] + classRecall[4] ))
classFmeasure[5] <- (2 * ( classPrecision[5] * classRecall[5]) / ( classPrecision[5] + classRecall[5] ))
print("Acccuracy for K-Nearest Neighbor for k = 7 knn(xtrain,xtest,ytrain,k=7,prob=TRUE)")

for (n in 1:5){
  if(is.nan(classRecall[n])) {
    classRecall[n] <- 0
  }
  if(is.nan(classPrecision[n])) {
    classPrecision[n] <- 0
  }
  if(is.nan(classFmeasure[n])) {
    classFmeasure[n] <- 0
  }
}

sumPrecision <- 0
sumRecall <- 0
sumAccuracy <- 0
sumFmeasure <- 0
for(n in 1:5){
  sumRecall <- sumRecall + classRecall[n]
  sumAccuracy <- sumAccuracy + classAccuracy[n]
  sumPrecision <- sumPrecision + classPrecision[n]
  sumFmeasure <- sumFmeasure + classFmeasure[n]
}
avgPrecision[1] = sumPrecision/5
avgAccuracy[1] = sumAccuracy/5
avgFmeasure[1] = sumFmeasure/5
avgRecall[1] = sumRecall/5
print(accuracy)
print(avgAccuracy)
print(avgPrecision)
print(avgRecall)
print(avgFmeasure)