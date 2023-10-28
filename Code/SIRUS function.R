
#Load required packages,
library(sirus)
library(caTools)

#A string representing the file path of the CSV file to be read.
dataset<-read.csv(file='D:\\One Drive\\OneDrive - Liverpool John Moores University\\Desktop\\ICIC\\Thesis\\Dataset\\Chapter 4\\CNvsMCI_afterFS.csv')


help(package="sirus")
#Split the dataset set into training and testing sets.
set.seed(42)
sample <- sample.split(dataset$Class, SplitRatio = 0.8)
train  <- subset(dataset, sample == TRUE)
test   <- subset(dataset, sample == FALSE)

#Convert each column of data frame to a factor.
train_x <- train[,1:(ncol(train)-1)]
for (i in colnames(train_x)){
  train_x[, i] <- as.factor(train_x[,i])}
train_y <- train[,"Class"]

test_x <- test[,1:(ncol(test)-1)]
for (i in colnames(test_x)){
  test_x[, i] <- as.factor(test_x[,i])}
test_y <- test[,"Class"]

#Estimate the optimal hyperparameter p0 used to select rules in 
#sirus.fit using cross-validation (Benard et al. 2021a, 2021b).
cv.grid <- sirus.cv(train_x, train_y, nfold = 10, ncv = 10, num.trees = 250)
print(cv.grid)
plot.error <- sirus.plot.cv(cv.grid,p0.criterion ="pred")$error
plot(plot.error)

#Fit SIRUS for a given p0 using  the 'X' data frame and 'target' variable 
sirus.m <- sirus.fit(train_x,train_y, p0=0.0395)
#Compute SIRUS predictions for new observations.
predictions <- sirus.predict(sirus.m, test_x)
preds_convert_to_integers <- as.integer(round(predictions))

#Test how the model is performancing on testing dataset
actual <- factor(test_y)
pred <- factor(preds_convert_to_integers)
library(caret)
confusionMatrix(pred, actual, mode = "everything", positive="1")

#Print the list of rules output by SIRUS.
sirus.print(sirus.m, digits = 5)


