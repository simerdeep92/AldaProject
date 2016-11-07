#install.packages("e1071")
library(caret)
library("e1071")
#csvData <- file.choose()
csvData <- read.csv("output.csv", header=T, sep=',')

#labelData <- file.choose()
labelData <- read.csv("Discussion_Category_Less5.csv", header=T, sep=',')

#labelList <- labelData[,c("Discussion.Category")]
#dim(labelData)
#dim(csvData)
#labelList <- labelList[]
labelList <- as.list(as.data.frame(t(labelData)))
integData <- cbind(csvData, labelData)
trainingDataSize <- floor(0.60 * nrow(integData))
set.seed(123)
train_ind <- sample(seq_len(nrow(integData)), size = trainingDataSize)
trainingData <- integData[train_ind, ]
testData <- integData[-train_ind, ]
xtrain = trainingData[,1:4023]
ytrain = trainingData[,4024]

xtest = testData[,1:4023]
ytest = testData[,4024]

tuneLinear = best.tune(svm,train.x=xtrain, train.y=ytrain,kernel ="linear")
print(summary(tuneLinear))
linearModel <- svm( xtrain, ytrain, kernel = "linear", type = 'C', cost = 1, gamma = 0.0002485707)

tunePolynomial = best.tune(svm,train.x=xtrain, train.y=ytrain,kernel ="polynomial")
print(summary(tunePolynomial))
polynomialModel <- svm( xtrain, ytrain, kernel = "polynomial", type = 'C', degree = 3, cost = 1, gamma = 0.0002485707)

tuneRadial = best.tune(svm,train.x=xtrain, train.y=ytrain,kernel ="radial")
print(summary(tuneRadial))
radialModel <- svm( xtrain, ytrain, kernel = "radial", type = 'C', cost = 1, gamma = 0.0002485707)

tuneSigmoid = best.tune(svm,train.x=xtrain, train.y=ytrain,kernel ="sigmoid")
print(summary(tuneSigmoid))
sigmoidModel <- svm( xtrain, ytrain, kernel = "sigmoid", type = 'C', cost = 1, gamma = 0.0002485707)

tuneQuadratic = best.tune(svm,train.x=xtrain, train.y=ytrain,kernel ="polynomial")
print(summary(tuneQuadratic))
quadraticModel <- svm( xtrain, ytrain, kernel = "polynomial", type = 'C', degree = 2, cost = 1, gamma = 0.0002485707)

methods = c("linear","polynomial","radial","sigmoid","quadratic");
precision = c()
recall=c()
FMeasure=c()
accuracy = c()

pred <- predict(linearModel,xtest)
confusionMatrix <- table(pred,ytest)
A <- confusionMatrix[1,1]
B <- confusionMatrix[1,2]
C <- confusionMatrix[2,1]
D <- confusionMatrix[2,2]
accuracy[1]= (A+D)/(A+B+C+D)
precision[1] <- A/(A+B)
recall[1] <- A/(A+C)
FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))


pred <- predict(polynomialModel,xtest)
confusionMatrix <- table(pred,ytest)
A <- confusionMatrix[1,1]
B <- confusionMatrix[1,2]
C <- confusionMatrix[2,1]
D <- confusionMatrix[2,2]
accuracy[2]= (A+D)/(A+B+C+D)
precision[2] <- A/(A+B)
recall[2] <- A/(A+C)
FMeasure[2] <-  (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))

pred <- predict(radialModel,xtest)
confusionMatrix <- table(pred,ytest)
A <- confusionMatrix[1,1]
B <- confusionMatrix[1,2]
C <- confusionMatrix[2,1]
D <- confusionMatrix[2,2]
accuracy[3]= (A+D)/(A+B+C+D)
precision[3] <- A/(A+B)
recall[3] <- A/(A+C)
FMeasure[3] <-  (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))


pred <- predict(sigmoidModel,xtest)
confusionMatrix <- table(pred,ytest)
A <- confusionMatrix[1,1]
B <- confusionMatrix[1,2]
C <- confusionMatrix[2,1]
D <- confusionMatrix[2,2]
accuracy[4]= (A+D)/(A+B+C+D)
precision[4] <- A/(A+B)
recall[4] <- A/(A+C)
FMeasure[4] <-  (2 * ( precision[4] * recall[4])/ ( precision[4] + recall[4] ))


pred <- predict(quadraticModel,xtest)
confusionMatrix <- table(pred,ytest)
A <- confusionMatrix[1,1]
B <- confusionMatrix[1,2]
C <- confusionMatrix[2,1]
D <- confusionMatrix[2,2]
accuracy[5]= (A+D)/(A+B+C+D)
precision[5] <- A/(A+B)
recall[5] <- A/(A+C)
FMeasure[5] <-  (2 * ( precision[5] * recall[5])/ ( precision[5] + recall[5] ))

resultTable = data.frame(methods,accuracy,precision,recall,FMeasure)

print(resultTable)