#ALL LIBRARIES MUST BE INSTALLED


#Import libraries
library(ggplot2)
library(data.table)
library(scatterplot3d)
library(tidyverse)
library(MLmetrics)
library(lubridate)
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%


options(max.print=1000000) #pour pas avoir de problème d'affichage (toutes les données seront affichées)

#lecture du fichier 
hcvdat <- read.csv("C:/Users/cherg/OneDrive/Bureau/hepatite/hcvdat0.csv", sep = ";")
print(hcvdat)
summary(hcvdat)



#DATA CLEANING

#Categorical variables are transformed into factors
hcvdat$Sex <- as.factor(hcvdat$Sex)
hcvdat$Category <- as.factor(hcvdat$Category)

#The Category variable of the original dataset can have the values:
levels(hcvdat$Category)


#Since the categories "0 = Blood Donor" and "0s = suspect Blood Donor" refer to people not infected by HCV, 
#these categories are changed to "Donor" , the rest of the categories are renamed as "Hepatitis" , "Fibrosis" , "Cirrhosis"
levels(hcvdat$Category) <- c("Donor", "Donor", "Hepatitis", "Fibrosis", "Cirrhosis")
hcvdat <- as.data.table(hcvdat)




#Data transformations
featureEngine <- function(data){
  dum_sex <- fastDummies::dummy_cols(data,select_columns = c("Sex"), remove_selected_columns    = TRUE)
  data_fe <- dum_sex %>% transmute(Category, 
                                   Age_m = Age*Sex_m, Age_f = Age*Sex_f,
                                   ALB_m = ALB*Sex_m, ALB_f = ALB*Sex_f,
                                   ALP_m = ALP*Sex_m, ALP_f = ALP*Sex_f,
                                   ALT_m = ALT*Sex_m, ALT_f = ALT*Sex_f,
                                   AST_m = AST*Sex_m, AST_f = AST*Sex_f,
                                   BIL_m = BIL*Sex_m, BIL_f = BIL*Sex_f,
                                   CHE_m = CHE*Sex_m, CHE_f = CHE*Sex_f,
                                   CHOL_m = CHOL*Sex_m, CHOL_f = CHOL*Sex_f,
                                   CREA_m = CREA*Sex_m, CREA_f = CREA*Sex_f,
                                   GGT_m = GGT*Sex_m, GGT_f = GGT*Sex_f,
                                   PROT_m = PROT*Sex_m, PROT_f = PROT*Sex_f)
  data_fe[is.na(data_fe)] <- 0
  
  return(as.data.table(data_fe))
}

hcvdat_fe <- featureEngine(hcvdat)

print(as.data.frame(hcvdat_fe))



#SVM Training and Prediction

library(caret)
set.seed(7)

#The dataset is divided into two, one for training and one for validation.
validationIndex <- caret::createDataPartition(hcvdat_fe$Category, p = 0.80, list = FALSE)

my_test  <- hcvdat_fe[-validationIndex,]
my_train <- hcvdat_fe[validationIndex,]

toHCV <- function(x){
  y <- plyr::revalue(x, c("Donor"="N", "Hepatitis"="Y", "Fibrosis"="Y", "Cirrhosis"="Y"))
  return(y)
}
hcv_test <- toHCV(my_test$Category)


#training
library(e1071)
svm_model <- svm(formula = Category~.,
                 data = my_train,
                 kernel = "polynomial", 
                 cost = 10, 
                 scale = FALSE, 
                 type = "C-classification")


#Prediction
svm_pred <- predict(svm_model, newdata = my_test)
hcv_svm <- toHCV(as.factor(svm_pred))

#Validation

#detection of infection
table(hcv_test, hcv_svm)

#the advance of the disease
table(my_test$Category, svm_pred)


#Precision

#Returns a data.frame where each category is exposed with its percentage of detection, 
#its percentage of success and its percentage of co-fusion (complementary to the success)
precisionMatrix <- function(y_true, y_pred){
  rnames <- levels(as.factor(y_true))

  #for each category the detection ratio is calculated and multiplied by 100
  detect <- mapply(function(x){return(100 * Recall(y_pred = y_pred, y_true = y_true, positive = x))}, rnames)
  
  #For each category the hit ratio is calculated and multiplied by 100
  success <- mapply(function(x){return(100 * Precision(y_pred = y_pred, y_true = y_true, positive = x))}, rnames)

  #Confusion is calculated
  confusion <- mapply(function(x){return (100 -x)}, success)
  
  df <- cbind.data.frame(as.data.frame(detect), as.data.frame(success), as.data.frame(confusion))
  names(df) <- c("Detection", "Success", "Confusion")
  return (df)
}

svm_pm <- rbind(precisionMatrix(hcv_test,hcv_svm), precisionMatrix(my_test$Category, svm_pred))
svm_pm <- svm_pm[rownames(svm_pm)!="Donor",]
svm_pm

cat("Accuracy:", Accuracy(y_pred = svm_pred, y_true = my_test$Category))


#Result with other kernels

#Radial basis
svm_model_rb <- svm(formula = Category~.,
                    data = my_train,
                    kernel = "radial", 
                    cost = 10, 
                    scale = FALSE, 
                    type = "C-classification")
svm_pred_rb <- predict(svm_model_rb, newdata = my_test)
cat("Accuracy:",
    MLmetrics::Accuracy(y_pred = svm_pred_rb, y_true = my_test$Category))


#Sigmoid
svm_model_sig <- svm(formula = Category~.,
                     data = my_train,
                     kernel = "sigmoid", 
                     cost = 10, 
                     scale = FALSE, 
                     type = "C-classification")
svm_pred_sig <- predict(svm_model_sig, newdata = my_test)
cat("Accuracy:",
    MLmetrics::Accuracy(y_pred = svm_pred_sig, y_true = my_test$Category))




#Ranger Forest Training and Prediction


#install.packages("ranger")
library(ranger)

#training
ranger_model <- ranger(
  Category~.,
  data = my_train,
  num.trees = 100,
  importance = 'impurity',
  write.forest = TRUE,
  min.node.size = 1,
  splitrule = "gini",
  verbose = TRUE,
  classification = TRUE
)

#Prediction
ranger_pred <- predict(ranger_model, data = my_test)
hcv_ranger <- toHCV(as.factor(ranger_pred$predictions))


#Validation
table(hcv_test, hcv_ranger)
table(my_test$Category, ranger_pred$predictions)


#Precision
ranger_pm <- rbind(precisionMatrix(hcv_test,hcv_ranger), precisionMatrix(my_test$Category, ranger_pred$predictions))
ranger_pm <- ranger_pm[rownames(ranger_pm)!="Donor",]
ranger_pm


cat("Accuracy:",
    MLmetrics::Accuracy(y_pred = ranger_pred$predictions, y_true = my_test$Category))
