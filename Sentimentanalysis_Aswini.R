# call libraries and set seed. 
library(readr)
library(caret)
library(plotly)
library(corrplot)
library(doParallel)
library(dplyr)
library(plotly)
library(tidyr)
library(corrplot)
library(ggplot2) 
set.seed(123)

#*****************************************************************
# set up parallel processing

library(doParallel)
# Find how many cores are on your machine
detectCores()  # result [8]
# Create Cluster with desired number of cores. 
cl <- makeCluster(4)
# Register Cluster
registerDoParallel(cl)
# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # result [4]

### -------------------import iphone dataset---------------------------------------------------------------
iphoneDF<- read_csv("smallmatrix_labeled_8d/iphone_smallmatrix_labeled_8d.csv")
# Check general data structure/info
str(iphoneDF)
summary(iphoneDF)
# Check all attributes of iphoneDF
names(iphoneDF)

#------------------------ check distribution of response variable        -------------------------------
plot_ly(iphoneDF, x= ~iphoneDF$iphonesentiment, type='histogram')  #### response variable sentiment score skewed to 5(positive review)

## Check for missing values
sum(is.na(iphoneDF))# no NA



# ========== Features Selection Methods  ---  preparing different data sets using features selection methods ====================
#====================    to find which method works best to apply on LARGEMATRIX  =====================



# -------------------------------   creating data set with nzv variables removed ----------------------------------------------=
#nearZeroVar() with saveMetrics = TRUE returns an object containing a table including: 
#frequency ratio, percentage unique, zero variance and near zero variance 

nzvMetrics <- nearZeroVar(iphoneDF, saveMetrics = TRUE)
nzvMetrics

# NearZeroVar() with saveMetrics = FALSE returns an vector
nzv <- nearZeroVar(iphoneDF, saveMetrics = FALSE) 
# Create a new data set and remove near zero variance features
iphoneNZV <- iphoneDF[,-nzv] # create new df with nzv variables removed
str(iphoneNZV)  #lists structure of the new df with nzv variables removed

#------------------------  creating data set with rfe features selection method    -------------------------------------------

# Let's sample the data before using RFE
iphoneSample <- iphoneDF[sample(1:nrow(iphoneDF), 1000, replace=FALSE),]
# Set up RFE Control with randomforest, repeated cross validation 
ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)
# Use RFE and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes= c(1:58), 
                  rfeControl= ctrl)

# Get results
rfeResults

## Plot results
plot(rfeResults, type=c("g", "o")) #optimal subset achieved with 18 features
#rferesults lists top 5 features that are important in prediction and removes features that are least important

## Create new data set with rfe recommended features
iphoneRFE <- iphoneDF[,predictors(rfeResults)]

## Add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphoneDF$iphonesentiment

#====================================  preprocess all above data sets prepared  ----------check na, set data types ==============================================-
# Factorize the dependent variable 
iphoneDF$iphonesentiment <- as.factor(iphoneDF$iphonesentiment)
iphoneNZV$iphonesentiment <- as.factor(iphoneNZV$iphonesentiment)
iphoneRFE$iphonesentiment <- as.factor(iphoneRFE$iphonesentiment)
str(iphoneDF$iphonesentiment)
str(iphoneDF)


#=================================#=================================#=================================#=================================
#=================================#=================================#=================================#=================================

#=======================  build models with original data set and find which model is the optimal to apply on other data sets --------------


# Define 70/30 train/test split from the original small matrix iphoneDF
inTraining <- createDataPartition(iphoneDF$iphonesentiment, p = .70, list = FALSE)
training <- iphoneDF[inTraining,]
testing <- iphoneDF[-inTraining,]

# 10 fold cross validation 
fitControl <- trainControl(method = "cv", number = 10)

#--------------- C5.0 ----------------------

C50 <- train(iphonesentiment~., data = training, method = "C5.0", trControl=fitControl)

prediction_C50 <- predict(C50, testing)

#--------------- RF ----------------------

rf <- train(iphonesentiment~., data = training, method = "rf", trControl=fitControl)

prediction_rf<- predict(rf, testing)

#--------------- SVM ----------------------

svm <- train(iphonesentiment~., data = training, method = "svmLinear", trControl=fitControl)

prediction_svm<- predict(svm, testing)

#------------------KKNN --------------------------------

kknn <- train(iphonesentiment~., data = training, method = "kknn", trControl=fitControl)

prediction_kknn<- predict(kknn, testing)

#============================================ check and compare performance metrics of above models  ===============================

postResample(prediction_C50, testing$iphonesentiment)

postResample(prediction_rf, testing$iphonesentiment)

postResample(prediction_svm, testing$iphonesentiment)

postResample(prediction_kknn, testing$iphonesentiment)

#====================================================================================================================================

# Create a confusion matrix from random forest predictions 
cmRF <- confusionMatrix(prediction_rf, testing$iphonesentiment) 
cmRF # accuracy 83% kappa 58%   # second time - acc 77% kappa 56%

cmc50 <- confusionMatrix(prediction_C50, testing$iphonesentiment) 
cmc50 #accuracy 77% kappa 56%      ################# choosing c50 for modeling because of less computing time   ###################

#=========================== Apply selected c50 from above trials to other df created earlier    ======================

#----------------------------- c50 on nzv df ------------------------------------------------

# Define train/test split of the iphoneNZV
inTraining_iphoneNZV <- createDataPartition(iphoneNZV$iphonesentiment, p = .70, list = FALSE)
training_NZV <- iphoneNZV[inTraining_iphoneNZV,]
testing_NZV <- iphoneNZV[-inTraining_iphoneNZV,]

# Apply c50 with 10-fold cross validation on iphoneNZV
c50_NZV <- train(iphonesentiment~., data = training_NZV, method = "C5.0", trControl=fitControl)

# Testing 
prediction_c50_NZV<- predict(c50_NZV, testing_NZV)
prediction_c50_NZV

#-------------  c50 on rfe df      -----------------------------------

# Define train/test split of the iphoneRFE
inTraining_iphoneRFE <- createDataPartition(iphoneRFE$iphonesentiment, p = .70, list = FALSE)
training_RFE <- iphoneRFE[inTraining_iphoneRFE,]
testing_RFE <- iphoneRFE[-inTraining_iphoneRFE,]

# Apply c50 with 10-fold cross validation on iphoneRFE
c50_RFE <- train(iphonesentiment~., data = training_RFE, method = "C5.0", trControl=fitControl)
# Testing 
prediction_c50_RFE<- predict(c50_RFE, testing_RFE)
prediction_c50_RFE

#-------------------------- Evaluate performance of above RF models  ---------------------------------------------------

postResample(prediction_c50_NZV, testing_NZV$iphonesentiment)
postResample(prediction_c50_RFE, testing_RFE$iphonesentiment) # accuracy and kappa are similar compared to predictions on original data set


#====================================================================================================================================
#============================ FEATURE ENGINEERING  ==============================================================================

#==================================== Alter & recode response variable factors - Feature Engineering ========================================

# Copy original data set into a new one that will be used to recode response variable
iphoneRC <- iphoneDF

# Recode sentiment and combine factor levels 0 & 1 and 4 & 5
iphoneRC$iphonesentiment <- recode(iphoneRC$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4)

# Make iphonesentiment a factor
iphoneRC$iphonesentiment <- as.factor(iphoneRC$iphonesentiment)

# Inspect the data structure of 'iphonesentiment'
str(iphoneRC$iphonesentiment)


#----------- try different recoding  -----------------------------------------------------------------------

iphoneRC1 <- iphoneDF
# Recode sentiment and combine factor levels 
iphoneRC1$iphonesentiment <- recode(iphoneRC1$iphonesentiment, '0' = 1, '1' = 2, '2' = 2, '3' = 3, '4' = 4, '5' = 4)

# Make iphonesentiment a factor
iphoneRC1$iphonesentiment <- as.factor(iphoneRC1$iphonesentiment)

# Inspect the data structure of 'iphonesentiment'
str(iphoneRC1$iphonesentiment)
str(iphoneRC1)

#---------------- build c50 model with altered response variable  - Recode Type 1  ------------------------------------

# Define train/test split of the iphoneRC 
inTrainingRC <- createDataPartition(iphoneRC$iphonesentiment, p = .70, list = FALSE)
training_RC <- iphoneRC[inTrainingRC,]
testing_RC <- iphoneRC[-inTrainingRC,]

# Use c50 with 10-fold CV on recoded sentiment levels data set 1
c50_RC <- train(iphonesentiment~., data = training_RC, method = "C5.0", trControl=fitControl)
# Testing 
prediction_c50_RC<- predict(c50_RC, testing_RC) 

# Evaluate the model
postResample(prediction_c50_RC, testing_RC$iphonesentiment)# acc = 84% kappa - 62%

cm_RC_iph<- confusionMatrix(prediction_c50_RC, testing_RC$iphonesentiment) 
cm_RC_iph

#---------------  c50 with  Different factor levels  - Recode Type 2   ==================================

# Define train/test split of the iphoneRC 
inTrainingRC1 <- createDataPartition(iphoneRC1$iphonesentiment, p = .70, list = FALSE)
training_RC1 <- iphoneRC1[inTrainingRC1,]
testing_RC1 <- iphoneRC1[-inTrainingRC1,]

# Use c50 with 10-fold CV on recoded sentiment levels data set 2
c50_RC1<- train(iphonesentiment~., data = training_RC1, method = "C5.0", trControl=fitControl)
# Testing 
prediction_c50_RC1<- predict(c50_RC1, testing_RC1) 

# Evaluate the model
postResample(prediction_c50_RC1, testing_RC1$iphonesentiment)

######################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# acc = 84% kappa - 69%
###################################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7


cm_RC1_iph<- confusionMatrix(prediction_c50_RC1, testing_RC1$iphonesentiment) 
cm_RC1_iph

#============================ PCA - FEATURE ENGINEERING ==================================================================

# original data from iphoneDF (no feature selection) 
# Excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(training[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)
## Created from 9083 samples and 58 variables
## 
## Pre-processing:
##   - centered (58)
##   - ignored (0)
##   - principal component signal extraction (58)
##   - scaled (58)
## 
## PCA needed 24 components to capture 95 percent of the variance

#--------- set different threshold ------------------
preprocessParams1 <- preProcess(training[,-59], method=c("center", "scale", "pca"), thresh = 0.85)
print(preprocessParams1)

# PCA needed 13 components to capture 85 percent of the variance

#--------------------------create train and test sets with pca parameters   ------------------------------------------------
# Use predict to apply pca parameters and create train.pca object from training set excluding dependent variable
train.pca <- predict(preprocessParams, training[,-59])

# Add the dependent variable to train.pca set
train.pca$iphonesentiment <- training$iphonesentiment

# Use predict to apply pca parameters and create test.pca object from testing set excluding dependent
test.pca <- predict(preprocessParams, testing[,-59])

# Add the dependent variable to testing set
test.pca$iphonesentiment <- testing$iphonesentiment
# inspect results
str(train.pca)
str(test.pca)

#----------------------   build c50 model with train.pca and test.pca set    --------------------------------------------------------

# 10 fold cross validation 
fitControl <- trainControl(method = "cv", number = 10)

# Build rf model with 10-fold CV and PCA train set  
c50_pca <- train(iphonesentiment~., data = train.pca, method = "C5.0", trControl=fitControl)

# predict using PCA test set 
prediction_c50_pca<- predict(c50_pca, test.pca)

# Evaluate the model
postResample(prediction_c50_pca, test.pca$iphonesentiment) #accuracy - 76% kappa - 53%  #accuracy has not improved



############################################################################################################################################
######################  Best learner model was RC1 recoded type 2  with accuracy 84% and kappa 69% ######################################### 

#=====================  SAMSUNG SMALL MATRIX =========================================================

# Importing SamsungDF 
samsungDF <- read_csv("smallmatrix_labeled_8d/galaxy_smallmatrix_labeled_9d.csv")

# Check general data structure/info
str(samsungDF)
summary(samsungDF)
# Check all attributes of iphoneDF
names(samsungDF)

#------------------------ check distribution of response variable        -------------------------------
plot_ly(samsungDF, x= ~samsungDF$galaxysentiment, type='histogram')  #### response variable sentiment score skewed to 5(positive review)

## Check for missing values
sum(is.na(iphoneDF))# no NA


# Create a new data set that will be used for recoding sentiment
samsungRC <- samsungDF

# Recode sentiment to combine factor levels 1 & 2 and 4 & 5
samsungRC$galaxysentiment <- recode(samsungRC$galaxysentiment, '0' = 1, '1' = 2, '2' = 2, '3' = 3, '4' = 4, '5' = 4)

# Make iphonesentiment a factor
samsungRC$galaxysentiment <- as.factor(samsungRC$galaxysentiment)
str(samsungRC)

# Define train/test split of the samsungRC
inTrainingRC_samsung <- createDataPartition(samsungRC$galaxysentiment, p = .70, list = FALSE)
training_RC_samsung <- samsungRC[inTrainingRC_samsung,]
testing_RC_samsung <- samsungRC[-inTrainingRC_samsung,]

# 10 fold cross validation 
fitControl <- trainControl(method = "cv", number = 10)

# Build c50 model
c50_RC_samsung <- train(galaxysentiment~., data = training_RC_samsung, method = "C5.0", trControl=fitControl)

# Testing 
prediction_c50_RC_samsung<- predict(c50_RC_samsung, testing_RC_samsung) 

# Evaluate the model 
postResample(prediction_c50_RC_samsung, testing_RC_samsung$galaxysentiment)  # accuracy - 84%  kappa - 59%

# Create a confusion matrix from random forest predictions 
cmc50_samsung <- confusionMatrix(prediction_c50_RC_samsung, testing_RC_samsung$galaxysentiment) 
cmc50_samsung

#================     predicting iphonesentiment on largematrix       =====================================================

iphoneLargeMatrix<- read_csv("C:/Users/karmeena/Downloads/combinedFile/combinedFile.csv")  #22095 * 59

# Remove web id column(first column) from the large matrix 
#iphoneLargeMatrix$id <- NULL

iphoneLargeMatrix$iphonesentiment <- as.factor(iphoneLargeMatrix$iphonesentiment)

#iphoneLargeMatrix$iphone <- as.numeric(iphoneLargeMatrix$iphone)
#iphoneLargeMatrix$samsunggalaxy <- as.numeric(iphoneLargeMatrix$samsunggalaxy)
#iphoneLargeMatrix$sonyxperia <- as.numeric(iphoneLargeMatrix$sonyxperia)


str(iphoneLargeMatrix)   # 22095 * 59
summary(iphoneLargeMatrix) #iphonesentiment all NA's
is.na(iphoneLargeMatrix)
sum(is.na(iphoneLargeMatrix)) #22095 ???


# Make predictions for iphonesentiment using rf recoded model
finalPred_iphone <- predict(c50_RC1, iphoneLargeMatrix)
summary(finalPred_iphone)

#> summary(finalPred_iphone)
#    1     2     3     4 
#  15437  832  996   4830 

#===========================predicting galaxysentiment on largematrix===========================================================

galaxylargematrix<- read_csv("C:/Users/karmeena/Downloads/combinedFile/combinedFile.csv")  #22095 * 59

# Remove web id column(first column) from the large matrix 
#iphoneLargeMatrix$id <- NULL

galaxylargematrix$galaxysentiment <- as.factor(galaxylargematrix$galaxysentiment)

#iphoneLargeMatrix$iphone <- as.numeric(iphoneLargeMatrix$iphone)
#iphoneLargeMatrix$samsunggalaxy <- as.numeric(iphoneLargeMatrix$samsunggalaxy)
#iphoneLargeMatrix$sonyxperia <- as.numeric(iphoneLargeMatrix$sonyxperia)


str(galaxylargematrix)   # 22095 * 59
summary(galaxylargematrix) #galaxysentiment all NA's
is.na(galaxylargematrix)
sum(is.na(galaxylargematrix)) #22095 ???

#na.omit(iphoneLargeMatrix)

# Make predictions for galaxysentiment using rf_samsung recoded model
finalPred_galaxy <- predict(c50_RC_samsung, galaxylargematrix)
summary(finalPred_galaxy)

#> summary(finalPred_iphone)
#    1     2     3     4 
#15567   796   1233  4499



stopCluster(cl)

