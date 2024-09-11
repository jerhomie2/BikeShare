library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(patchwork)
library(vroom)

train <- vroom("./BikeShare/train.csv") # like read.csv ish but faster
test <- vroom("./BikeShare/test.csv")

#----- EDA -----

glimpse(train) # lists variable type of each column
plot_intro(train)
plot_correlation(train)

temp <- ggplot(train, 
               mapping = aes(x = temp, y = count)) +
  geom_point() +
  geom_smooth(se = F)

weather <- ggplot(train, 
                  mapping = aes(x = weather)) +
  geom_bar()

season <- ggplot(train, 
                 mapping = aes(x = season)) +
  geom_bar()

humidity <- ggplot(train,
                   mapping = aes(x=humidity, y = count)) +
  geom_point() +
  geom_smooth(se = F)

(temp + weather) / (season + humidity)

#-----Linear Regression-----

## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>% # Regression just means quantitative response
  fit(formula=count~season + holiday + workingday +weather+temp+atemp+humidity+windspeed, data=train)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=test) # Use fit to predict
bike_predictions ## Look at the output

## Format the Predictions for Submission to Kaggle1
kaggle_submission <- bike_predictions %>%
bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./BikeShare/LinearPreds.csv", delim=",")
