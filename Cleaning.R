library(tidyverse)
library(tidymodels)
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
  step_mutate(datetime_hour = as.factor(date)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_rm(c(datetime)) %>%
  step_corr(all_predictors(), threshold = 0.85)
prepped_recipe <- prep(bike_recipe)
baked_train <- bake(prepped_recipe, new_data=train) 
baked_test <- bake(prepped_recipe, new_data=test)

lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

bike_workflow <- workflow() %>% 
  add_recipe(bike_recipe) %>%
  add_model(lin_model) %>%
  fit(data = train)

lin_preds <- exp(predict(bike_workflow, new_data = test))

kaggle_submission <- lin_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./BikeShare/CleaningPreds.csv", delim=",")

