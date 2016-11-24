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
ctrl <- trainControl(method = "cv", savePred=T, classProb=T)

train_control <- trainControl(method="cv", number=10)
# fix the parameters of the algorithm
grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE))
# train the model
model <- train(labelList ~., data=csvData, trControl=train_control, method="nb", tuneGrid=grid)
# summarize results
print(model)


#radialModel <- svm(csvData, labelData, type = 'C', kernel = "radial", cost = 1, gamma = 0.015625, trControl = ctrl)
#print(radialModel)