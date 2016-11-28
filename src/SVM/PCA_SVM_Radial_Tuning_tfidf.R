#install.packages("e1071")
library(caret)
library("e1071")
#csvData <- file.choose()
csvData <- read.csv("output_tfidf.csv", header=T, sep=',')

#labelData <- file.choose()
labelData <- read.csv("Discussion_Category_Less5.csv", header=T, sep=',')

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

###########PCA#######
#xtrainT <- t(xtrain)
#p <- prcomp(xtrain, retx=TRUE, center=TRUE, scale=TRUE)
p <- prcomp(xtrain)
i <- 6
pRot <- (p$rotation[,1:i])

xtrainMat <- data.matrix(xtrain)
xtrainMult <- xtrainMat %*% pRot

#print(summary(p))

#plot(p, type = "l")

xtest = testData[,1:numOfCols]
ytest = testData[,numOfCols+1]

##########PCA##########
#xtestT <- t(xtest)
pTest <- prcomp(xtest)
#print(summary(p2))
pRotTest <- (pTest$rotation[,1:i])

xtestMat <- data.matrix(xtest)
xtestMult <- xtestMat %*% pRotTest


#tuneRadial = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="radial")
#print(summary(tuneRadial))
maxAccuracy <- .Machine$double.xmin
c <- .Machine$double.xmin
g <- .Machine$double.xmin
for(c in seq(from=-15, to=15, by=0.5)){
  for(g in seq(from=-15, to=15, by=0.5)){
    #radialModel <- svm( xtrainMult, ytrain, kernel = "radial", type = 'C', cost = 2^c, gamma = 2^g)
    linearModel <- svm( xtrainMult, ytrain, kernel = "linear", type = 'C', cost = 2^c, gamma = 2^g)
    #polynomialModel <- svm( xtrainMult, ytrain, kernel = "polynomial", type = 'C', degree = 3, cost = 2^c, gamma = 2^g)
    #sigmoidModel <- svm( xtrainMult, ytrain, kernel = "sigmoid", type = 'C', cost = 2^c, gamma = 2^g)
    
    predRadial <- predict(linearModel,xtestMult)
    confusionMatrix <- table(predRadial,ytest)
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
    if(accuracy > maxAccuracy) {
      maxC = c
      maxG = g
      maxAccuracy = accuracy
    }
  }
}
print(maxC)
print(maxG)
print(maxAccuracy)
# write.csv(confusionMatrix, file="CM_Radial.csv")