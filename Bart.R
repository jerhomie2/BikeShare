library(tidyverse)
library(tidymodels)
library(rpart)
library(vroom)

train <- vroom("./BikeShare/train.csv") # like read.csv ish but faster
test <- vroom("./BikeShare/test.csv")

train <- train %>%
  subset(select = -c(casual, registered)) %>%
  mutate(count = log(count))

bike_recipe <- recipe(count~., data=train) %>% # Set model formula and dataset2
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = as.factor(weather)) %>%
  step_date(datetime, features = "dow") %>%
  step_time(datetime, features = c("hour")) %>%
  step_date(datetime, features = c("month")) %>%
  step_date(datetime, features = c("year")) %>%
  step_mutate(datetime_dow = as.factor(datetime_dow)) %>%
  step_mutate(datetime_hour = as.factor(datetime_hour)) %>%
  step_mutate(datetime_month = as.factor(datetime_month)) %>%
  step_mutate(datetime_year = as.factor(datetime_year)) %>%
  step_interact(~datetime_hour:workingday) %>%
  step_interact(~datetime_hour:datetime_dow) %>%
  step_mutate(season = as.factor(season)) %>%
  step_zv(all_predictors()) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1
prepped_recipe <- prep(bike_recipe)
baked_train <- bake(prepped_recipe, new_data=train) 
baked_test <- bake(prepped_recipe, new_data=test)

my_mod <- bart(trees=1000) %>% #Type of model
  set_engine("dbarts") %>% # What R function to use
  set_mode("regression")

## Set workflow
bart_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod)

## Finalize workflow & fit it
final_wf <-
  bart_wf %>%
  fit(data = train)

## Predict
lin_preds <-final_wf %>% 
  predict(new_data = test) %>%
  exp()

kaggle_submission <- lin_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./BikeShare/BartPreds.csv", delim=",")

