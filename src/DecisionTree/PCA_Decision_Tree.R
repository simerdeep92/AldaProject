library(tree)
library(caret)
library("e1071")
library(gridExtra)
library(grid)

#csvData <- file.choose()
csvData <- read.csv("C:\\Users\\Bita\\Desktop\\AldaProject\\AldaProject\\src\\SVM\\output_tfidf.csv", header=T, sep=',')

#labelData <- file.choose()
labelData <- read.csv("C:\\Users\\Bita\\Desktop\\AldaProject\\AldaProject\\src\\SVM\\Discussion_Category_Less5_2.csv", header=T, sep=',')

labelList <- as.list(as.data.frame(t(labelData)))
integData <- cbind(csvData, labelData)
trainingDataSize <- floor(0.70 * nrow(integData))
set.seed(123)
train_ind <- sample(seq_len(nrow(integData)), size = trainingDataSize)
trainingData <- integData[train_ind, ]
trainLabel<-labelData[train_ind,]
testData <- integData[-train_ind, ]
testLabel<-labelData[-train_ind,]
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



#Fit the tree model using training data
xtrainMultLabeled=data.frame(xtrainMult,trainLabel)
tree_model=tree(trainLabel~.,as.data.frame(xtrainMultLabeled))
plot(tree_model)
text(tree_model,pretty=0)

# Do the prediction
xtestMultLabeled=data.frame(xtestMult,testLabel)
tree_pred=predict(tree_model,as.data.frame(xtestMultLabeled),type="class")
100*(1-mean(tree_pred!=ytest))
#Calculating the accuracy:
  
#Do cross validation with pruning
set.seed(2)
cv_tree=cv.tree(tree_model,,prune.misclass)

names(cv_tree)
plot(cv_tree$size,cv_tree$dev,type="b")

#Prune with the best size 
pruned_model=prune.misclass(tree_model,best=10)
plot(pruned_model)
                                             

##Cheque how it is doing
tree_predict=predict(pruned_model,as.data.frame(xtestMult),type="class")
100*(1-mean(tree_predict!=ytest))
