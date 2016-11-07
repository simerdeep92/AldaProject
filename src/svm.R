#install.packages("e1071")
library(caret)
library("e1071")
#csvData <- file.choose()
csvData <- read.csv("output2.csv", header=T, sep=',')

#labelData <- file.choose()
labelData <- read.csv("Discussion_Category_Less5_1.csv", header=T, sep=',')

#labelList <- labelData[,c("Discussion.Category")]
#dim(labelData)
#dim(csvData)
#labelList <- labelList[]
labelList <- as.list(as.data.frame(t(labelData)))


radialModel <- svm(csvData[1:369,], labelData[1:369,], type = 'C', kernel = "radial", cost = 1, gamma = 0.015625)
pred <- predict(radialModel, csvData[370:399,])
print(pred)