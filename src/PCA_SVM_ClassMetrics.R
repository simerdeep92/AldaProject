#install.packages("e1071")
library(caret)
library("e1071")
#csvData <- file.choose()
csvData <- read.csv("output.csv", header=T, sep=',')

#labelData <- file.choose()
labelData <- read.csv("Discussion_Category_Less5_2.csv", header=T, sep=',')

#labelList <- labelData[,c("Discussion.Category")]
#dim(labelData)
#dim(csvData)
#labelList <- labelList[]
labelList <- as.list(as.data.frame(t(labelData)))
integData <- cbind(csvData, labelData)
trainingDataSize <- floor(0.70 * nrow(integData))
set.seed(123)
train_ind <- sample(seq_len(nrow(integData)), size = trainingDataSize)
trainingData <- integData[train_ind, ]
testData <- integData[-train_ind, ]
xtrain = trainingData[,1:4023]
ytrain = trainingData[,4024]

###########PCA#######
#xtrainT <- t(xtrain)
#p <- prcomp(xtrain, retx=TRUE, center=TRUE, scale=TRUE)
p <- prcomp(xtrain)
pRot <- (p$rotation[,1:31])

xtrainMat <- data.matrix(xtrain)
xtrainMult <- xtrainMat %*% pRot

#print(summary(p))

#plot(p, type = "l")

##########PCA in caret##########
# trans = preProcess(xtrain,
#                      method=c("BoxCox", "center",
#                               "scale", "pca"))

xtest = testData[,1:4023]
ytest = testData[,4024]

##########PCA##########
#xtestT <- t(xtest)
pTest <- prcomp(xtest)
#print(summary(p2))
pRotTest <- (pTest$rotation[,1:31])

xtestMat <- data.matrix(xtest)
xtestMult <- xtestMat %*% pRotTest

tuneLinear = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="linear")
#print(summary(tuneLinear))
linearModel <- svm( xtrainMult, ytrain, kernel = "linear", type = 'C', cost = 1, gamma = 0.03225806)

tunePolynomial = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="polynomial")
#print(summary(tunePolynomial))
polynomialModel <- svm( xtrainMult, ytrain, kernel = "polynomial", type = 'C', degree = 3, cost = 1, gamma = 0.03225806)

tuneRadial = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="radial")
#print(summary(tuneRadial))
radialModel <- svm( xtrainMult, ytrain, kernel = "radial", type = 'C', cost = 1, gamma = 0.03225806)

tuneSigmoid = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="sigmoid")
#print(summary(tuneSigmoid))
sigmoidModel <- svm( xtrainMult, ytrain, kernel = "sigmoid", type = 'C', cost = 1, gamma = 0.03225806)

tuneQuadratic = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="polynomial")
#print(summary(tuneQuadratic))
quadraticModel <- svm( xtrainMult, ytrain, kernel = "polynomial", type = 'C', degree = 3, cost = 1, gamma = 0.03225806)

methods = c("linear","polynomial","radial","sigmoid","quadratic");

precision = c()
recall=c()
FMeasure=c()
accuracy = c()
classAccuracy = matrix(, nrow = 5, ncol = 5)
classPrecision = matrix(, nrow = 5, ncol = 5)
classRecall = matrix(, nrow = 5, ncol = 5)

pred <- predict(linearModel,xtestMult)
confusionMatrix <- table(pred,ytest)
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
grid.table(confusionMatrix)
sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy[1]= (A1+B2+C3+D4+E5)/sumOfElements
classAccuracy[1,1] = A1/sumOfElements
classAccuracy[1,2] = B2/sumOfElements
classAccuracy[1,3] = C3/sumOfElements
classAccuracy[1,4] = D4/sumOfElements
classAccuracy[1,5] = E5/sumOfElements

classPrecision[1,1] = A1/(A1+B1+C1+D1+E1)
classPrecision[1,2] = B2/(A1+B1+C1+D1+E1)
classPrecision[1,3] = C3/(A1+B1+C1+D1+E1)
classPrecision[1,4] = D4/(A1+B1+C1+D1+E1)
classPrecision[1,5] = E5/(A1+B1+C1+D1+E1)

classRecall[1,1] = A1/(A1+A2+A3+A4+A5)
classRecall[1,2] = B2/(A1+A2+A3+A4+A5)
classRecall[1,3] = C3/(A1+A2+A3+A4+A5)
classRecall[1,4] = D4/(A1+A2+A3+A4+A5)
classRecall[1,5] = E5/(A1+A2+A3+A4+A5)
#precision[1] <- A1/(A1+B1+C1+D1+E1)
#recall[1] <- A1/(A1+A2+A3+A4+A5)
#FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))


predPolynomial <- predict(polynomialModel,xtestMult)
confusionMatrix <- table(predPolynomial,ytest)
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
dev.off()
grid.table(confusionMatrix)
sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy[2]= (A1+B2+C3+D4+E5)/sumOfElements
classAccuracy[2,1] = A1/sumOfElements
classAccuracy[2,2] = B2/sumOfElements
classAccuracy[2,3] = C3/sumOfElements
classAccuracy[2,4] = D4/sumOfElements
classAccuracy[2,5] = E5/sumOfElements

classPrecision[2,1] = A1/(A1+B1+C1+D1+E1)
classPrecision[2,2] = B2/(A1+B1+C1+D1+E1)
classPrecision[2,3] = C3/(A1+B1+C1+D1+E1)
classPrecision[2,4] = D4/(A1+B1+C1+D1+E1)
classPrecision[2,5] = E5/(A1+B1+C1+D1+E1)

classRecall[2,1] = A1/(A1+A2+A3+A4+A5)
classRecall[2,2] = B2/(A1+A2+A3+A4+A5)
classRecall[2,3] = C3/(A1+A2+A3+A4+A5)
classRecall[2,4] = D4/(A1+A2+A3+A4+A5)
classRecall[2,5] = E5/(A1+A2+A3+A4+A5)
# #precision[1] <- A1/(A1+B1+C1+D1+E1)
# #recall[1] <- A1/(A1+A2+A3+A4+A5)
# #FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))
#
predRadial <- predict(radialModel,xtestMult)
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
dev.off()
grid.table(confusionMatrix)
sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy[3]= (A1+B2+C3+D4+E5)/sumOfElements
classAccuracy[3,1] = A1/sumOfElements
classAccuracy[3,2] = B2/sumOfElements
classAccuracy[3,3] = C3/sumOfElements
classAccuracy[3,4] = D4/sumOfElements
classAccuracy[3,5] = E5/sumOfElements

classPrecision[3,1] = A1/(A1+B1+C1+D1+E1)
classPrecision[3,2] = B2/(A1+B1+C1+D1+E1)
classPrecision[3,3] = C3/(A1+B1+C1+D1+E1)
classPrecision[3,4] = D4/(A1+B1+C1+D1+E1)
classPrecision[3,5] = E5/(A1+B1+C1+D1+E1)

classRecall[3,1] = A1/(A1+A2+A3+A4+A5)
classRecall[3,2] = B2/(A1+A2+A3+A4+A5)
classRecall[3,3] = C3/(A1+A2+A3+A4+A5)
classRecall[3,4] = D4/(A1+A2+A3+A4+A5)
classRecall[3,5] = E5/(A1+A2+A3+A4+A5)
# #precision[1] <- A1/(A1+B1+C1+D1+E1)
# #recall[1] <- A1/(A1+A2+A3+A4+A5)
# #FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))
#
predSigmoid <- predict(sigmoidModel,xtestMult)
confusionMatrix <- table(predSigmoid,ytest)
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
dev.off()
grid.table(confusionMatrix)
sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy[4]= (A1+B2+C3+D4+E5)/sumOfElements
classAccuracy[4,1] = A1/sumOfElements
classAccuracy[4,2] = B2/sumOfElements
classAccuracy[4,3] = C3/sumOfElements
classAccuracy[4,4] = D4/sumOfElements
classAccuracy[4,5] = E5/sumOfElements

classPrecision[4,1] = A1/(A1+B1+C1+D1+E1)
classPrecision[4,2] = B2/(A1+B1+C1+D1+E1)
classPrecision[4,3] = C3/(A1+B1+C1+D1+E1)
classPrecision[4,4] = D4/(A1+B1+C1+D1+E1)
classPrecision[4,5] = E5/(A1+B1+C1+D1+E1)

classRecall[4,1] = A1/(A1+A2+A3+A4+A5)
classRecall[4,2] = B2/(A1+A2+A3+A4+A5)
classRecall[4,3] = C3/(A1+A2+A3+A4+A5)
classRecall[4,4] = D4/(A1+A2+A3+A4+A5)
classRecall[4,5] = E5/(A1+A2+A3+A4+A5)
# #precision[1] <- A1/(A1+B1+C1+D1+E1)
# #recall[1] <- A1/(A1+A2+A3+A4+A5)
# #FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))

predQuadratic <- predict(quadraticModel,xtestMult)
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
dev.off()
grid.table(confusionMatrix)
sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy[5]= (A1+B2+C3+D4+E5)/sumOfElements
classAccuracy[5,1] = A1/sumOfElements
classAccuracy[5,2] = B2/sumOfElements
classAccuracy[5,3] = C3/sumOfElements
classAccuracy[5,4] = D4/sumOfElements
classAccuracy[5,5] = E5/sumOfElements

classPrecision[5,1] = A1/(A1+B1+C1+D1+E1)
classPrecision[5,2] = B2/(A1+B1+C1+D1+E1)
classPrecision[5,3] = C3/(A1+B1+C1+D1+E1)
classPrecision[5,4] = D4/(A1+B1+C1+D1+E1)
classPrecision[5,5] = E5/(A1+B1+C1+D1+E1)

classRecall[5,1] = A1/(A1+A2+A3+A4+A5)
classRecall[5,2] = B2/(A1+A2+A3+A4+A5)
classRecall[5,3] = C3/(A1+A2+A3+A4+A5)
classRecall[5,4] = D4/(A1+A2+A3+A4+A5)
classRecall[5,5] = E5/(A1+A2+A3+A4+A5)
#precision[1] <- A1/(A1+B1+C1+D1+E1)
#recall[1] <- A1/(A1+A2+A3+A4+A5)
#FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))
resultTable = data.frame(methods,accuracy)
dev.off()
print(resultTable)
colnames(classAccuracy) <- c("A", "B", "C", "D", "E")
rownames(classAccuracy) <- c("A", "B", "C", "D", "E")

colnames(classPrecision) <- c("A", "B", "C", "D", "E")
rownames(classPrecision) <- c("A", "B", "C", "D", "E")

colnames(classRecall) <- c("A", "B", "C", "D", "E")
rownames(classRecall) <- c("A", "B", "C", "D", "E")