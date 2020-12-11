# The dataset includes information about:
# 1- Customers who left within the last month
# 2- Services that each customer has signed up for: multiple lines, Internet, online security, online backup,
# device protection, tech support, and streaming TV and movies
# 3- customer account information: How long they're been a customer, contract, payment, method, 
# paperless billing, monthly charges, and total charges
# 4- Demographic info about customers: gender, age, range, and if they have partners and dependents

################################################################################
### Example on how to use Keras to develop accurate deep learning model in R
################################################################################
# use recipes package for preprocessing workflow
# Explain the ANN with "lime package"
# cross check the Lime results with Correlation analysis using the "corrr package"

# packages used:
# - keras
# - Lime
# - Tidyquant
# - Tidyverse
# - rsample
# - recipes
# - yardstick
# - corrr

pkgs <- c("keras","lime","tidyquant", "rsample", "recipes", "yardstick", "corrr")
install.packages(pkgs)

# Load libraries
library(keras)
library(lime)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)

install_keras()

################################################################################
### Import the data
################################################################################

churn_data_raw <- read.csv("Telco-Customer-Churn.csv")
glimpse(churn_data_raw)

################################################################################
### Preprocess Data
################################################################################

# Prune the data
# - remove customerID column
# - drop the na value (only 11 na / small percentage of the total population)

indx <- apply(churn_data_raw, 2, function(x) any(is.na(x)))
table(indx)
indexes <- which(is.na(churn_data_raw$TotalCharges))
indexes
churn_data_raw <- churn_data_raw[- c(indexes), ]

churn_data_tbl <- churn_data_raw %>%
  select(-customerID) %>%
  select(Churn, everything())

glimpse(churn_data_tbl)
any(is.na(churn_data_tbl)) # no NA values

# Split test/training data
set.seed(100)
train_test_split <- initial_split(churn_data_tbl, prop=0.8)
train_test_split

#train_data
train_tbl <- training(train_test_split)
test_tbl <- testing(train_test_split)

library(ggplot2)
ggplot(churn_data_tbl[1:length(churn_data_tbl),], aes(x=tenure)) +
  geom_histogram()+
  ggtitle("Tenure Counts")+
  xlab("Tenure (months)")+
  ylab("Count")

library(ggplot2)
ggplot(churn_data_tbl[1:length(churn_data_tbl),], aes(x=tenure)) +
  #geom_histogram(binwidth = 8)+
  geom_density(alpha=0.2, fill="#FF6666")+
  ggtitle("Tenure Counts")+
  xlab("Tenure (months)")+
  ylab("Count")

library(ggplot2)
ggplot(churn_data_tbl[1:length(churn_data_tbl),], aes(x=TotalCharges)) +
  #geom_histogram(binwidth = 8)+
  geom_density(alpha=0.2, fill="#FF6666")+
  ggtitle("Total charges Counts")+
  xlab("Total Charges")+
  ylab("Count")

library(ggplot2)
ggplot(churn_data_tbl[1:length(churn_data_tbl),], aes(x=log(TotalCharges, base = exp(1)))) +
  #geom_histogram(binwidth = 8)+
  geom_density(alpha=0.2, fill="#FF6666")+
  ggtitle("Log(Total charges)")+
  xlab("log(Total Charges)")+
  ylab("Count")


# determine if log transformation improves correlation
# between TotalCharges and Churn

logTotalCharges <- as.numeric(log(train_tbl$TotalCharges, base=exp(1)))
TotalCharges <- as.numeric(train_tbl$TotalCharges)
churn_num = ifelse(train_tbl$Churn=="No",0,1)

m_data <- matrix(c(logTotalCharges,TotalCharges,churn_num), ncol=3)

library("Hmisc")
correlation_m_data <- rcorr(m_data, type = c("pearson","spearman"))

cc <- as.matrix(correlation$r)
rownames(cc) <- c("logTotalCharges","TotalCharges","churn_num")
colnames(cc) <- c("logTotalCharges","TotalCharges","churn_num")
cc

# log total charges has higher correlation with Churn - using Log Total charges is more useful for the predictive model

unique(train_tbl$Contract)
unique(train_tbl$InternetService)
unique(train_tbl$MultipleLines)
unique(train_tbl$PaymentMethod)



library(ggpubr)
contract <- ggplot(train_tbl[1:length(train_tbl),], aes(x = Contract)) +
  geom_bar()+ 
  xlab("Contract type")+
  ylab("Count")

InternetService <- ggplot(train_tbl[1:length(train_tbl),], aes(x = InternetService)) +
  geom_bar(fill="#FF6666")+ 
  xlab("InternetService type")+
  ylab("Count")

MultipleLines <- ggplot(train_tbl[1:length(train_tbl),], aes(x = MultipleLines)) +
  geom_bar(fill="orange")+ 
  xlab("MultipleLines type")+
  ylab("Count")

PaymentMethod <- ggplot(train_tbl[1:length(train_tbl),], aes(x = PaymentMethod)) +
  geom_bar(fill="Blue")+ 
  xlab("PaymentMethod type")+
  ylab("Count")

ggarrange(contract, InternetService, MultipleLines, PaymentMethod, 
          labels = c("A", "B", "C", "D"),
          ncol = 2, nrow = 2)


################################################################################
##                   Preprocessing with Recipes
################################################################################
# using:
# - step_discretize() --> Discretize continuous variable to group costumers
# - step_log()        --> Apply log to Variable to improve correlation
# - step_center()     --> One-hot encode the categorical data. Add columns of 
#                         one/zero for categorical data with 3 or more categories
# - step_scale()      --> Mean center the data
# Create recipe       --> Scale the data

rec_obj <- recipe(Churn~., data = train_tbl) %>%
  step_discretize(tenure, options=list(cuts=6)) %>%
  step_log(TotalCharges) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep (data = train_tbl)

rec_obj

# Baking with your recipe

# predictors
x_train_tbl <- bake(rec_obj, new_data = train_tbl) %>% select(-Churn)
x_test_tbl <- bake(rec_obj, new_data = test_tbl) %>% select(-Churn)

glimpse(x_train_tbl)

# Response variable for training and testing sets
y_train_vec <- ifelse(pull(train_tbl, Churn)=="Yes", 1, 0) # pull is similar to $
y_test_vec <- ifelse(pull(test_tbl, Churn)=="Yes",1 ,0)


################################################################################
###          Model Customer Churn with Keras (Deep Learning)
################################################################################

# Building MLP (Multiple-Layer Perceptron)
model_keras <- keras_model_sequential()
model_keras %>%
  # First layer
  layer_dense(units = 16, kernel_initializer = "uniform", activation = "relu", input_shape = ncol(x_train_tbl)) %>%
  layer_dropout(rate=0.1) %>%
  # Second hidden layer
  layer_dense(units = 16, kernel_initializer = "uniform", activation = "relu") %>%
  layer_dropout(rate=0.1) %>%
  # Output layer
  layer_dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid") %>%
summary(model_keras)

model_keras %>%
  compile(optimizer = "adam", loss = "binary_crossentropy", metrics = c('accuracy'))

history <- fit(
  object = model_keras,
  x = as.matrix(x_train_tbl),
  y = y_train_vec,
  batch_size = 50,
  epochs =35,
  validation_split = 0.3
)

print(history)


# Prediction using the test dataset

# Predicted class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()
# Predicted Class Probability
yhat_keras_prob_vec <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# inspect Performance with Yardstick

estimates_keras_tbl <- data.frame(
  truth = as.factor(ifelse(y_test_vec == "1", "Yes", "No")),
  estimate = as.factor(ifelse(yhat_keras_class_vec == 1, "Yes", "No")),
  class_prob = yhat_keras_prob_vec
)

options(yardstick.event_first = FALSE)

# Confusion Matrix
library(caret)
ctable <- confusionMatrix(estimates_keras_tbl$truth, estimates_keras_tbl$estimate)
ctable
fourfoldplot(ctable$table, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

prec_model <- precision(estimates_keras_tbl$truth, estimates_keras_tbl$estimate)
prec_model

recall_model <- recall(estimates_keras_tbl$truth, estimates_keras_tbl$estimate)
recall_model

F_meas(estimates_keras_tbl$truth, estimates_keras_tbl$estimate, beta = 1)

