#install.packages("e1071")
library(caret)
library("e1071")
library(gridExtra)
library(grid)

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

###########PCA#######
#xtrainT <- t(xtrain)
#p <- prcomp(xtrain, retx=TRUE, center=TRUE, scale=TRUE)
p <- prcomp(xtrain)

# Choosing 6 dimesions after experimenting in PCA_SVM_ClassMetrics_Test.R
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

# c and g values received after experimenting in PCA_SVM_Radial_Tuning_tfidf.R
c = 2^3
g = 2^(-4)
tuneLinear = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="linear")
#print(summary(tuneLinear))
linearModel <- svm( xtrainMult, ytrain, kernel = "linear", type = 'C', cost = c, gamma = g)

tunePolynomial = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="polynomial")
#print(summary(tunePolynomial))
polynomialModel <- svm( xtrainMult, ytrain, kernel = "polynomial", type = 'C', degree = 3, cost = c, gamma = g)

tuneRadial = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="radial")
#print(summary(tuneRadial))
radialModel <- svm( xtrainMult, ytrain, kernel = "radial", type = 'C', cost = c, gamma = g)

tuneSigmoid = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="sigmoid")
#print(summary(tuneSigmoid))
sigmoidModel <- svm( xtrainMult, ytrain, kernel = "sigmoid", type = 'C', cost = c, gamma = g)

tuneQuadratic = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="polynomial")
#print(summary(tuneQuadratic))
quadraticModel <- svm( xtrainMult, ytrain, kernel = "polynomial", type = 'C', degree = 3, cost = c, gamma = g)

methods = c("LinearSVM", "PolynomialSVM", "RadialSVM", "SigmoidSVM", "QuadraticSVM")

precision = c()
recall=c()
FMeasure=c()
accuracy = c()
classAccuracy = matrix(nrow = 5, ncol = 5)
classPrecision = matrix(nrow = 5, ncol = 5)
classRecall = matrix(nrow = 5, ncol = 5)

pred <- predict(linearModel,xtestMult)
confusionMatrix1 <- table(pred,ytest)
A1 <- confusionMatrix1[1,1]
A2 <- confusionMatrix1[1,2]
A3 <- confusionMatrix1[1,3]
A4 <- confusionMatrix1[1,4]
A5 <- confusionMatrix1[1,5]
B1 <- confusionMatrix1[2,1]
B2 <- confusionMatrix1[2,2]
B3 <- confusionMatrix1[2,3]
B4 <- confusionMatrix1[2,4]
B5 <- confusionMatrix1[2,5]
C1 <- confusionMatrix1[3,1]
C2 <- confusionMatrix1[3,2]
C3 <- confusionMatrix1[3,3]
C4 <- confusionMatrix1[3,4]
C5 <- confusionMatrix1[3,5]
D1 <- confusionMatrix1[4,1]
D2 <- confusionMatrix1[4,2]
D3 <- confusionMatrix1[4,3]
D4 <- confusionMatrix1[4,4]
D5 <- confusionMatrix1[4,5]
E1 <- confusionMatrix1[5,1]
E2 <- confusionMatrix1[5,2]
E3 <- confusionMatrix1[5,3]
E4 <- confusionMatrix1[5,4]
E5 <- confusionMatrix1[5,5]
# grid.table(confusionMatrix)
sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy[1]= (A1+B2+C3+D4+E5)/sumOfElements
classAccuracy[1,1] = A1/sumOfElements
classAccuracy[1,2] = B2/sumOfElements
classAccuracy[1,3] = C3/sumOfElements
classAccuracy[1,4] = D4/sumOfElements
classAccuracy[1,5] = E5/sumOfElements

classPrecision[1,1] = A1/(A1+B1+C1+D1+E1)
classPrecision[1,2] = B2/(A2+B2+C2+D2+E2)
classPrecision[1,3] = C3/(A3+B3+C3+D3+E3)
classPrecision[1,4] = D4/(A4+B4+C4+D4+E4)
classPrecision[1,5] = E5/(A5+B5+C5+D5+E5)

classRecall[1,1] = A1/(A1+A2+A3+A4+A5)
classRecall[1,2] = B2/(B1+B2+B3+B4+B5)
classRecall[1,3] = C3/(C1+C2+C3+C4+C5)
classRecall[1,4] = D4/(D1+D2+D3+D4+D5)
classRecall[1,5] = E5/(E1+E2+E3+E4+E5)
#precision[1] <- A1/(A1+B1+C1+D1+E1)
#recall[1] <- A1/(A1+A2+A3+A4+A5)
#FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))

predPolynomial <- predict(polynomialModel,xtestMult)
confusionMatrix2 <- table(predPolynomial,ytest)
A1 <- confusionMatrix2[1,1]
A2 <- confusionMatrix2[1,2]
A3 <- confusionMatrix2[1,3]
A4 <- confusionMatrix2[1,4]
A5 <- confusionMatrix2[1,5]
B1 <- confusionMatrix2[2,1]
B2 <- confusionMatrix2[2,2]
B3 <- confusionMatrix2[2,3]
B4 <- confusionMatrix2[2,4]
B5 <- confusionMatrix2[2,5]
C1 <- confusionMatrix2[3,1]
C2 <- confusionMatrix2[3,2]
C3 <- confusionMatrix2[3,3]
C4 <- confusionMatrix2[3,4]
C5 <- confusionMatrix2[3,5]
D1 <- confusionMatrix2[4,1]
D2 <- confusionMatrix2[4,2]
D3 <- confusionMatrix2[4,3]
D4 <- confusionMatrix2[4,4]
D5 <- confusionMatrix2[4,5]
E1 <- confusionMatrix2[5,1]
E2 <- confusionMatrix2[5,2]
E3 <- confusionMatrix2[5,3]
E4 <- confusionMatrix2[5,4]
E5 <- confusionMatrix2[5,5]
# dev.off()
# grid.table(confusionMatrix)
sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy[2]= (A1+B2+C3+D4+E5)/sumOfElements
classAccuracy[2,1] = A1/sumOfElements
classAccuracy[2,2] = B2/sumOfElements
classAccuracy[2,3] = C3/sumOfElements
classAccuracy[2,4] = D4/sumOfElements
classAccuracy[2,5] = E5/sumOfElements

classPrecision[2,1] = A1/(A1+B1+C1+D1+E1)
classPrecision[2,2] = B2/(A2+B2+C2+D2+E2)
classPrecision[2,3] = C3/(A3+B3+C3+D3+E3)
classPrecision[2,4] = D4/(A4+B4+C4+D4+E4)
classPrecision[2,5] = E5/(A5+B5+C5+D5+E5)

classRecall[2,1] = A1/(A1+A2+A3+A4+A5)
classRecall[2,2] = B2/(B1+B2+B3+B4+B5)
classRecall[2,3] = C3/(C1+C2+C3+C4+C5)
classRecall[2,4] = D4/(D1+D2+D3+D4+D5)
classRecall[2,5] = E5/(E1+E2+E3+E4+E5)
# #precision[1] <- A1/(A1+B1+C1+D1+E1)
# #recall[1] <- A1/(A1+A2+A3+A4+A5)
# #FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))
#
predRadial <- predict(radialModel,xtestMult)
confusionMatrix3 <- table(predRadial,ytest)
A1 <- confusionMatrix3[1,1]
A2 <- confusionMatrix3[1,2]
A3 <- confusionMatrix3[1,3]
A4 <- confusionMatrix3[1,4]
A5 <- confusionMatrix3[1,5]
B1 <- confusionMatrix3[2,1]
B2 <- confusionMatrix3[2,2]
B3 <- confusionMatrix3[2,3]
B4 <- confusionMatrix3[2,4]
B5 <- confusionMatrix3[2,5]
C1 <- confusionMatrix3[3,1]
C2 <- confusionMatrix3[3,2]
C3 <- confusionMatrix3[3,3]
C4 <- confusionMatrix3[3,4]
C5 <- confusionMatrix3[3,5]
D1 <- confusionMatrix3[4,1]
D2 <- confusionMatrix3[4,2]
D3 <- confusionMatrix3[4,3]
D4 <- confusionMatrix3[4,4]
D5 <- confusionMatrix3[4,5]
E1 <- confusionMatrix3[5,1]
E2 <- confusionMatrix3[5,2]
E3 <- confusionMatrix3[5,3]
E4 <- confusionMatrix3[5,4]
E5 <- confusionMatrix3[5,5]
# dev.off()
# grid.table(confusionMatrix)
sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy[3]= (A1+B2+C3+D4+E5)/sumOfElements
classAccuracy[3,1] = A1/sumOfElements
classAccuracy[3,2] = B2/sumOfElements
classAccuracy[3,3] = C3/sumOfElements
classAccuracy[3,4] = D4/sumOfElements
classAccuracy[3,5] = E5/sumOfElements

classPrecision[3,1] = A1/(A1+B1+C1+D1+E1)
classPrecision[3,2] = B2/(A2+B2+C2+D2+E2)
classPrecision[3,3] = C3/(A3+B3+C3+D3+E3)
classPrecision[3,4] = D4/(A4+B4+C4+D4+E4)
classPrecision[3,5] = E5/(A5+B5+C5+D5+E5)

classRecall[3,1] = A1/(A1+A2+A3+A4+A5)
classRecall[3,2] = B2/(B1+B2+B3+B4+B5)
classRecall[3,3] = C3/(C1+C2+C3+C4+C5)
classRecall[3,4] = D4/(D1+D2+D3+D4+D5)
classRecall[3,5] = E5/(E1+E2+E3+E4+E5)
# #precision[1] <- A1/(A1+B1+C1+D1+E1)
# #recall[1] <- A1/(A1+A2+A3+A4+A5)
# #FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))
#
predSigmoid <- predict(sigmoidModel,xtestMult)
confusionMatrix4 <- table(predSigmoid,ytest)
A1 <- confusionMatrix4[1,1]
A2 <- confusionMatrix4[1,2]
A3 <- confusionMatrix4[1,3]
A4 <- confusionMatrix4[1,4]
A5 <- confusionMatrix4[1,5]
B1 <- confusionMatrix4[2,1]
B2 <- confusionMatrix4[2,2]
B3 <- confusionMatrix4[2,3]
B4 <- confusionMatrix4[2,4]
B5 <- confusionMatrix4[2,5]
C1 <- confusionMatrix4[3,1]
C2 <- confusionMatrix4[3,2]
C3 <- confusionMatrix4[3,3]
C4 <- confusionMatrix4[3,4]
C5 <- confusionMatrix4[3,5]
D1 <- confusionMatrix4[4,1]
D2 <- confusionMatrix4[4,2]
D3 <- confusionMatrix4[4,3]
D4 <- confusionMatrix4[4,4]
D5 <- confusionMatrix4[4,5]
E1 <- confusionMatrix4[5,1]
E2 <- confusionMatrix4[5,2]
E3 <- confusionMatrix4[5,3]
E4 <- confusionMatrix4[5,4]
E5 <- confusionMatrix4[5,5]
# dev.off()
# grid.table(confusionMatrix)
sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy[4]= (A1+B2+C3+D4+E5)/sumOfElements
classAccuracy[4,1] = A1/sumOfElements
classAccuracy[4,2] = B2/sumOfElements
classAccuracy[4,3] = C3/sumOfElements
classAccuracy[4,4] = D4/sumOfElements
classAccuracy[4,5] = E5/sumOfElements


classPrecision[4,1] = A1/(A1+B1+C1+D1+E1)
classPrecision[4,2] = B2/(A2+B2+C2+D2+E2)
classPrecision[4,3] = C3/(A3+B3+C3+D3+E3)
classPrecision[4,4] = D4/(A4+B4+C4+D4+E4)
classPrecision[4,5] = E5/(A5+B5+C5+D5+E5)

classRecall[4,1] = A1/(A1+A2+A3+A4+A5)
classRecall[4,2] = B2/(B1+B2+B3+B4+B5)
classRecall[4,3] = C3/(C1+C2+C3+C4+C5)
classRecall[4,4] = D4/(D1+D2+D3+D4+D5)
classRecall[4,5] = E5/(E1+E2+E3+E4+E5)
# #precision[1] <- A1/(A1+B1+C1+D1+E1)
# #recall[1] <- A1/(A1+A2+A3+A4+A5)
# #FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))

predQuadratic <- predict(quadraticModel,xtestMult)
confusionMatrix5 <- table(predQuadratic,ytest)
A1 <- confusionMatrix5[1,1]
A2 <- confusionMatrix5[1,2]
A3 <- confusionMatrix5[1,3]
A4 <- confusionMatrix5[1,4]
A5 <- confusionMatrix5[1,5]
B1 <- confusionMatrix5[2,1]
B2 <- confusionMatrix5[2,2]
B3 <- confusionMatrix5[2,3]
B4 <- confusionMatrix5[2,4]
B5 <- confusionMatrix5[2,5]
C1 <- confusionMatrix5[3,1]
C2 <- confusionMatrix5[3,2]
C3 <- confusionMatrix5[3,3]
C4 <- confusionMatrix5[3,4]
C5 <- confusionMatrix5[3,5]
D1 <- confusionMatrix5[4,1]
D2 <- confusionMatrix5[4,2]
D3 <- confusionMatrix5[4,3]
D4 <- confusionMatrix5[4,4]
D5 <- confusionMatrix5[4,5]
E1 <- confusionMatrix5[5,1]
E2 <- confusionMatrix5[5,2]
E3 <- confusionMatrix5[5,3]
E4 <- confusionMatrix5[5,4]
E5 <- confusionMatrix5[5,5]
# dev.off()
# grid.table(confusionMatrix)
sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
accuracy[5]= (A1+B2+C3+D4+E5)/sumOfElements
classAccuracy[5,1] = A1/sumOfElements
classAccuracy[5,2] = B2/sumOfElements
classAccuracy[5,3] = C3/sumOfElements
classAccuracy[5,4] = D4/sumOfElements
classAccuracy[5,5] = E5/sumOfElements

classPrecision[5,1] = A1/(A1+B1+C1+D1+E1)
classPrecision[5,2] = B2/(A2+B2+C2+D2+E2)
classPrecision[5,3] = C3/(A3+B3+C3+D3+E3)
classPrecision[5,4] = D4/(A4+B4+C4+D4+E4)
classPrecision[5,5] = E5/(A5+B5+C5+D5+E5)

classRecall[5,1] = A1/(A1+A2+A3+A4+A5)
classRecall[5,2] = B2/(B1+B2+B3+B4+B5)
classRecall[5,3] = C3/(C1+C2+C3+C4+C5)
classRecall[5,4] = D4/(D1+D2+D3+D4+D5)
classRecall[5,5] = E5/(E1+E2+E3+E4+E5)
#precision[1] <- A1/(A1+B1+C1+D1+E1)
#recall[1] <- A1/(A1+A2+A3+A4+A5)
#FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))
resultTable = data.frame(methods,accuracy)
# dev.off()
print(resultTable)
colnames(classAccuracy) <- c("A", "B", "C", "D", "E")
rownames(classAccuracy) <- c("LinearSVM", "PolynomialSVM", "RadialSVM", "SigmoidSVM", "QuadraticSVM")

colnames(classPrecision) <- c("A", "B", "C", "D", "E")
rownames(classPrecision) <- c("LinearSVM", "PolynomialSVM", "RadialSVM", "SigmoidSVM", "QuadraticSVM")

colnames(classRecall) <- c("A", "B", "C", "D", "E")
rownames(classRecall) <- c("LinearSVM", "PolynomialSVM", "RadialSVM", "SigmoidSVM", "QuadraticSVM")
