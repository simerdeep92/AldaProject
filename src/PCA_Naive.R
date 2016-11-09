#install.packages("e1071")
library(caret)
library("e1071")
library("klaR")
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
ytrain <- as.factor(ytrain)

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

#########PCA using caret#############
# trans = preProcess(xtest, 
#                      method=c("BoxCox", "center", 
#                               "scale", "pca"))
# p2 = predict(trans, xtest)

xtrainMultData <- as.data.frame(xtrainMult)


method = c("naiveBayes");
#tuneLinear = best.tune(svm,train.x=xtrain, train.y=ytrain,kernel ="linear")
#print(summary(tuneLinear))
linearModel <- naiveBayes(ytrain ~., data=xtrainMultData)

predQuadratic <- predict(linearModel, xtestMult)
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

#precision[1] <- A1/(A1+B1+C1+D1+E1)
#recall[1] <- A1/(A1+A2+A3+A4+A5)
#FMeasure[1] <- (2 * ( precision[1] * recall[1])/ ( precision[1] + recall[1] ))
resultTable = data.frame(method,accuracy)

print(resultTable)

