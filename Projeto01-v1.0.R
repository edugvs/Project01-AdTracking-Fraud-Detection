# Building a Predictive Model for Detecting Click Traffic Fraud in Mobile Application Advertisements


## 1 - Business problem ##
# In this project, we will build a machine learning model to determine whether a click is
# fraudulent or not. For the construction of this project, we will use the R language.

# The dataset is available on Kaggle at the link:
#https: //www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data


## 2 - Loading data ##

# Importing the required libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(dplyr)
library(lubridate)
library(fasttime)
library(Amelia)
library(caret)
library(randomForest)
library(smotefamily)
library(scales)
library(imbalance)
library(caTools)
library(pROC)


# Loading train data
df_original <- fread('train_sample.csv')

# Copying dataset
data <- df_original



## Data Dictionary ##
# According to the documentation, each line of training data contains a click record and the following variables:

#Data fields:
#ip: ip address of click.
#app: app id for marketing.
#device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
#os: os version id of user mobile phone
#channel: channel id of mobile ad publisher
#click_time: timestamp of click (UTC)
#attributed_time: if user download the app for after clicking an ad, this is the time of the app download
#is_attributed: the target that is to be predicted, indicating the app was downloaded

#Note that ip, app, device, os, and channel are encoded


## 3 - Data munging ##

# Transforming variables types
# Converting click_time to date type
data$click_time <- fastPOSIXct(data$click_time)

# Converting attributed_time to date type
data$attributed_time <- fastPOSIXct(data$attributed_time)


# Checking for missing values 
missmap(data, 
        main = "Training Data - Missing Values Map", 
        col = c("yellow", "black"), 
        legend = FALSE)

# There are several missing values in the variable "attributed_time", as most records users have NOT downloaded
# of the app, the time has not been recorded and the column does not fill with any value

# Checking for duplicate records
table(duplicated(data))

# Eliminating duplicate records from the dataset.
data <- data[!duplicated(data), ]



## 4 - Exploratory data analysis ##
# Checking the dataset
glimpse(data)


# Checking the unique values for each variable
plot_var <- c("os", "channel", "device", "app", "attributed_time", "click_time", "ip")
data[, lapply(.SD, uniqueN), .SDcols = plot_var] %>%
  melt(variable.name = "features", value.name = "unique_values") %>%
  ggplot(aes(reorder(features, -unique_values), unique_values)) +
  geom_bar(stat = "identity", fill = "steelblue") + 
  scale_y_log10(breaks = c(50,100,250, 500, 10000, 50000)) +
  geom_text(aes(label = unique_values), vjust = 1.6, color = "white", size=3.5) +
  theme_minimal() +
  labs(x = "features", y = "Number of unique values")


# Analysing target variable
table(data$is_attributed)

data %>%
  mutate(Downloaded = factor(is_attributed, labels = c('No', 'Yes')))  %>%
  ggplot(aes(x = Downloaded, fill = Downloaded)) + 
  geom_bar() + 
  labs(title = 'Var Target (is_attributed) Balancing') + 
  ylab('rows') + 
  theme_minimal()

#We can see that the class is unbalanced so we must correct this before running the predictive modeling


# Exploring categorical variables
data[, .N, by = os][order(-N)][1:10] %>% 
  ggplot(aes(reorder(os, -N), N)) +
  geom_bar(stat="identity", fill="steelblue") + 
  theme_minimal() +
  geom_text(aes(label = round(N / sum(N), 2)), vjust = 1.6, color = "white", size=2.5) +
  labs(x = "os") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

data[, .N, by = channel][order(-N)][1:10] %>% 
  ggplot(aes(reorder(channel, -N), N)) +
  geom_bar(stat="identity", fill="steelblue") + 
  theme_minimal() +
  geom_text(aes(label = round(N / sum(N), 2)), vjust = 1.6, color = "white", size=2.5) +
  labs(x = "channel") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

data[, .N, by = device][order(-N)][1:10] %>% 
  ggplot(aes(reorder(device, -N), N)) +
  geom_bar(stat="identity", fill="steelblue") + 
  theme_minimal() +
  geom_text(aes(label = round(N / sum(N), 2)), vjust = 1.6, color = "white", size=2.5) +
  labs(x = "device") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

data[, .N, by = app][order(-N)][1:10] %>% 
  ggplot(aes(reorder(app, -N), N)) +
  geom_bar(stat="identity", fill="steelblue") + 
  theme_minimal() +
  geom_text(aes(label = round(N / sum(N), 2)), vjust = 1.6, color = "white", size=2.5) +
  labs(x = "app") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))    


# Time Series Analysis
summary(data$click_time)

# Number of clicks that resulted in a download
data %>%
  mutate(dates = floor_date(click_time, unit = 'hour')) %>%
  group_by(dates) %>%
  summarise(downloads = sum(as.numeric(is_attributed==1))) %>%
  ggplot(aes(x = dates, y = downloads)) +
  geom_line() +
  scale_x_datetime(date_breaks = '2 hours', date_labels = '%d %b - %H') +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) + 
  xlab('Time') +
  ylab('Total Downloads') +
  labs(title = 'Downloads per hour')

# Number of clicks that did NOT result in a download
data %>%
  mutate(dates = floor_date(click_time, unit = 'hour')) %>%
  group_by(dates) %>%
  summarise(nodownloads = sum(!is_attributed)) %>%
  ggplot(aes(x = dates, y = nodownloads)) +
  geom_line() +
  scale_x_datetime(date_breaks = '2 hours', date_labels = '%d %b - %H') +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) + 
  xlab('Time') +
  ylab('Unrealized downloads') +
  labs(title = 'Unrealized downloads per hours')


# Clearly, we can see a pattern in the time that downloads are made, with values well below the average between 5 pm-7pm


# Checking the difference, in seconds, between the click on the ad and the time that the download was performed
secs_diff <- difftime(data$attributed_time, data$click_time, units="secs")
secs_diff <- na.omit(secs_diff)
summary(as.numeric(secs_diff))



## 5 - Creating the Predictive Model ##

# Feature Engineering
# Let's split click_date variable into "day" and "hour" and add the variable "secs_diff"
train_data <- data %>%  
  mutate(day = day(click_time), hour = hour(click_time), min = minute(click_time), sec = second(click_time)) %>% 
  select(-c(click_time, attributed_time))

# Converting the target varbiable to factor
train_data$is_attributed <- as.factor(train_data$is_attributed)
str(train_data)

# Normalizing data
dados_norm <- as.data.frame(lapply(train_data[, -6], rescale))
train_data <- cbind(train_data[,6], dados_norm)

# Class Balancing
new.sample <- rwo(train_data, numInstances = 99545, classAttr = "is_attributed")
train_data <- rbind(train_data, new.sample)

table(new.sample$is_attributed)
table(train_data$is_attributed)

# Split data into train and test datasets
split = sample.split(train_data$is_attributed, SplitRatio = 0.70)
train = subset(train_data, split==TRUE)
test = subset(train_data, split==FALSE)

str(train)
str(test)

# Checking the importance of variables using RandomForrest
# Define a seed to allow the same experiment results to be reproducible
set.seed(100)
    
model1 <- randomForest(is_attributed ~ .,
                      data       = train, 
                      ntree      = 15, 
                      nodesize   = 2, 
                      importance = T)


# Print model
model1

# Plotting a graph to visualize the level of importance of each variable in the prediction process of the target variable.
plot_rf <- as.data.frame(varImpPlot(model1))

# Sorting the name of the variables in descending order of importance.
names <- rownames(plot_rf[order(plot_rf$MeanDecreaseAccuracy, decreasing = T),])

# Print result
plot_rf[order(plot_rf$MeanDecreaseAccuracy, decreasing = T),]

# Making predictions
pred1 <- predict(model1, test)

# Computes a confusion matrix generated from the model created
caret::confusionMatrix(test$is_attributed, pred1, positive = '1')

# More details about the model
summary(model1)

# Calculating the AUC (Area Under the Curve) for the model
rfAUC <- auc(roc(as.integer(test$is_attributed), as.integer(pred1)))
rfAUC


# Building the model with the most relevant variables
# Defining formula to be used by the model.
f <- is_attributed ~ device + app + channel + ip + os

# Creating the final model based on the Random Forest algorithm
model_rf <- randomForest(f, 
                         ntree      = 15, 
                         nodesize   = 2, 
                         data       = train)

# Making predictions with the model based on the new Random Forest algorithm
pred2 <- predict(model_rf, test, type = 'response')

# Creating the Confusion Matrix  
caret::confusionMatrix(test$is_attributed, pred2, positive = '1')

# There was no significant improvement in the accuracy of the model after the selection of the most important variables

# Calculating the AUC (Area Under the Curve) for the model
rfAUC2 <- auc(roc(as.integer(test$is_attributed), as.integer(pred2)))
rfAUC2












