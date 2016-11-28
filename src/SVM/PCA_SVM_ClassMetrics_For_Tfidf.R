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
modelList = c()
tuneLinear = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="linear")
#print(summary(tuneLinear))
linearModel <- svm( xtrainMult, ytrain, kernel = "linear", type = 'C', cost = c, gamma = g)
modelList <- c(modelList, linearModel)

tunePolynomial = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="polynomial")
#print(summary(tunePolynomial))
polynomialModel <- svm( xtrainMult, ytrain, kernel = "polynomial", type = 'C', degree = 3, cost = c, gamma = g)
modelList <- c(modelList, polynomialModel)

tuneRadial = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="radial")
#print(summary(tuneRadial))
radialModel <- svm( xtrainMult, ytrain, kernel = "radial", type = 'C', cost = c, gamma = g)
modelList <- c(modelList, radialModel)

tuneSigmoid = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="sigmoid")
#print(summary(tuneSigmoid))
sigmoidModel <- svm( xtrainMult, ytrain, kernel = "sigmoid", type = 'C', cost = c, gamma = g)
modelList <- c(modelList, sigmoidModel)

tuneQuadratic = best.tune(svm,train.x=xtrainMult, train.y=ytrain,kernel ="polynomial")
#print(summary(tuneQuadratic))
quadraticModel <- svm( xtrainMult, ytrain, kernel = "polynomial", type = 'C', degree = 3, cost = c, gamma = g)
modelList <- c(modelList, quadraticModel)

methods = c("LinearSVM", "PolynomialSVM", "RadialSVM", "SigmoidSVM", "QuadraticSVM")

precision = c()
recall=c()
FMeasure=c()
accuracy = c()
classAccuracy = matrix(nrow = 5, ncol = 5)
classPrecision = matrix(nrow = 5, ncol = 5)
classRecall = matrix(nrow = 5, ncol = 5)
classFmeasure = matrix(nrow = 5, ncol = 5)
avgPrecision = c()
avgRecall = c()
avgAccuracy = c()
avgFmeasure = c()

count = 0
for (eachMethod in methods) {
  count = count + 1
  if (eachMethod == "LinearSVM"){
    pred <- predict(linearModel,xtestMult)
  }
  else if(eachMethod == "PolynomialSVM"){
    pred <- predict(polynomialModel, xtestMult)
  }
  else if(eachMethod == "RadialSVM"){
    pred <- predict(radialModel, xtestMult)
  }
  else if(eachMethod == "SigmoidSVM"){
    pred <- predict(polynomialModel, xtestMult)
  }
  else if(eachMethod == "QuadraticSVM"){
    pred <- predict(quadraticModel, xtestMult)
  }
  confusionMatrix <- matrix(ncol=5, nrow=5)
  confusionMatrix <- table("Predictions"=pred, Actual=ytest)
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
  # grid.table(confusionMatrix)
  sumOfElements <- A1+A2+A3+A4+A5+B1+B2+B3+B4+B5+C1+C2+C3+C4+C5+D1+D2+D3+D4+D5+E1+E2+E3+E4+E5
  accuracy[count]= (A1+B2+C3+D4+E5)/sumOfElements
  classAccuracy[count,1] = A1/sumOfElements
  classAccuracy[count,2] = B2/sumOfElements
  classAccuracy[count,3] = C3/sumOfElements
  classAccuracy[count,4] = D4/sumOfElements
  classAccuracy[count,5] = E5/sumOfElements
  
  classPrecision[count,1] = A1/(A1+B1+C1+D1+E1)
  classPrecision[count,2] = B2/(A2+B2+C2+D2+E2)
  classPrecision[count,3] = C3/(A3+B3+C3+D3+E3)
  classPrecision[count,4] = D4/(A4+B4+C4+D4+E4)
  classPrecision[count,5] = E5/(A5+B5+C5+D5+E5)
  
  classRecall[count,1] = A1/(A1+A2+A3+A4+A5)
  classRecall[count,2] = B2/(B1+B2+B3+B4+B5)
  classRecall[count,3] = C3/(C1+C2+C3+C4+C5)
  classRecall[count,4] = D4/(D1+D2+D3+D4+D5)
  classRecall[count,5] = E5/(E1+E2+E3+E4+E5)
  
  classFmeasure[count,1] <- (2 * ( classPrecision[count,1] * classRecall[count,1]) / ( classPrecision[count,1] + classRecall[count,1] ))
  classFmeasure[count,2] <- (2 * ( classPrecision[count,2] * classRecall[count,2]) / ( classPrecision[count,2] + classRecall[count,2] ))
  classFmeasure[count,3] <- (2 * ( classPrecision[count,3] * classRecall[count,3]) / ( classPrecision[count,3] + classRecall[count,3] ))
  classFmeasure[count,4] <- (2 * ( classPrecision[count,4] * classRecall[count,4]) / ( classPrecision[count,4] + classRecall[count,4] ))
  classFmeasure[count,5] <- (2 * ( classPrecision[count,5] * classRecall[count,5]) / ( classPrecision[count,5] + classRecall[count,5] ))
}

for (m in 1:5){
  for (n in 1:5){
    if(is.nan(classRecall[m, n])) {
      classRecall[m, n] <- 0
    }
    if(is.nan(classPrecision[m, n])) {
      classPrecision[m, n] <- 0
    }
    if(is.nan(classFmeasure[m, n])) {
      classFmeasure[m, n] <- 0
    }
  }
}

resultTable = data.frame(methods,accuracy)
# dev.off()
print(resultTable)

for (m in 1:5){
  sumPrecision <- 0
  sumRecall <- 0
  sumAccuracy <- 0
  sumFmeasure <- 0
  for(n in 1:5){
    sumRecall <- sumRecall + classRecall[m, n]
    sumAccuracy <- sumAccuracy + classAccuracy[m, n]
    sumPrecision <- sumPrecision + classPrecision[m, n]
    sumFmeasure <- sumFmeasure + classFmeasure[m, n]
  }
  avgPrecision[m] = sumPrecision/5
  avgAccuracy[m] = sumAccuracy/5
  avgFmeasure[m] = sumFmeasure/5
  avgRecall[m] = sumRecall/5
}

colnames(classAccuracy) <- c("A", "B", "C", "D", "E")
rownames(classAccuracy) <- c("LinearSVM", "PolynomialSVM", "RadialSVM", "SigmoidSVM", "QuadraticSVM")

colnames(classPrecision) <- c("A", "B", "C", "D", "E")
rownames(classPrecision) <- c("LinearSVM", "PolynomialSVM", "RadialSVM", "SigmoidSVM", "QuadraticSVM")

colnames(classRecall) <- c("A", "B", "C", "D", "E")
rownames(classRecall) <- c("LinearSVM", "PolynomialSVM", "RadialSVM", "SigmoidSVM", "QuadraticSVM")

colnames(classFmeasure) <- c("A", "B", "C", "D", "E")
rownames(classFmeasure) <- c("LinearSVM", "PolynomialSVM", "RadialSVM", "SigmoidSVM", "QuadraticSVM")

resultTableAvgAccuracy = data.frame(methods, avgAccuracy)
resultTableAvgRecall = data.frame(methods, avgRecall)
resultTableAvgFmeasure = data.frame(methods, avgFmeasure)
resultTableAvgPrecision = data.frame(methods, avgPrecision)