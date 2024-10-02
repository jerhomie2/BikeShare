library(tidyverse)
library(tidymodels)
library(rpart)
library(vroom)
library(stacks)

train <- vroom("./BikeShare/train.csv") 
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
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1
prepped_recipe <- prep(bike_recipe)
baked_train <- bake(prepped_recipe, new_data=train) 
baked_test <- bake(prepped_recipe, new_data=test)

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Penalized regression workflow
preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5) ## L^2 total tuning possibilities

## Run the CV
preg_models <- preg_wf %>%
tune_grid(resamples=folds,
          grid=preg_tuning_grid,
          metrics=metric_set(rmse, mae),
          control = untunedModel) # including the control grid in the tuning ensures you can
                                  # call on it later in the stacked model

## Random forest model
forest_mod <- rand_forest(mtry=tune(),
                      min_n=tune(),
                      trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

## Set workflow
forest_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(forest_mod)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(mtry(range = c(1,50)),
                                      min_n(),
                                      levels = 5)

## Run the CV
forest_mods <- forest_wf %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse, mae),
            control = untunedModel)

## Specify which models to include
my_stack <- stacks() %>%
  add_candidates(preg_models) %>%
  add_candidates(forest_mods)

## Fit the stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset

## If you want to build your own metalearner you'll have to do so manually
## using
stackData <- as_tibble(my_stack)

## Use the stacked data to get a prediction
preds <- stack_mod %>% 
  predict(new_data=test) %>%
  exp()

## Kaggle Submission
kaggle_submission <- preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./BikeShare/Stack.csv", delim=",")





