# install.packages("e1071")
# install.packages('MASS')
library(MASS)
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

print(trainingData)
###########PCA#######
l <- lda(trainingData[,numOfCols+1] ~ trainingData[,numOfCols], data = trainingData)
print(l)