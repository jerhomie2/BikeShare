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
  step_mutate(datetime_hour = as.factor(datetime_hour)) %>%
  step_mutate(datetime_month = as.factor(datetime_month)) %>%
  step_interact(~datetime_hour:workingday) %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_rm(c(datetime)) %>%
  step_corr(all_predictors(), threshold = 0.85) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables7
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1
prepped_recipe <- prep(bike_recipe)
baked_train <- bake(prepped_recipe, new_data=train) 
baked_test <- bake(prepped_recipe, new_data=test)

my_mod <- boost_tree(mtry = tune(),
                     min_n=tune(),
                     trees=50) %>% #Type of model
  set_engine("xgboost") %>% # What R function to use
  set_mode("regression") %>%
  translate()

## Set workflow
boost_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(mtry(range = c(1,50)),
                                      min_n(),
                                      levels = 5)

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse, mae))

## Plot Results
collect_metrics(CV_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(data = ., aes(x = penalty, y = mean, color = factor(mixture))) +
  geom_line()

## Find best tuning params
bestTune <- CV_results %>%
  select_best(metric = "rmse")

## Finalize workflow & fit it
final_wf <-
  boost_wf %>%
  finalize_workflow(bestTune) %>%
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
vroom_write(x=kaggle_submission, file="./BikeShare/BoostPreds.csv", delim=",")

