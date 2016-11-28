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

method = c("logistic regression");
#tuneLinear = best.tune(svm,train.x=xtrain, train.y=ytrain,kernel ="linear")
#print(summary(tuneLinear))
#linearModel <- naiveBayes(ytrain ~., data=xtrainMultData)
xtrainMultDataFrame <- as.data.frame(xtrainMultData)
linearModel <- glm(ytrain ~., family=binomial(link='logit'), data=xtrainMultDataFrame)

xtestMultDataFrame <- as.data.frame(xtestMult)

predQuadratic <- predict(linearModel, xtestMultDataFrame)
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
