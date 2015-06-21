# Machine Learning Class Project
Derek Franks  
June 20, 2015  


#Introduction
The purpose of this data analysis is to use data collected from accellerometers located on the belt, forearm, arm, and dumbell of 6 participants to predict the type of exercise they were engaged in.

The training data set consists of 19622 observations of 160 variables, including a "target" variable, `classe` that identifies the type of exercise.  It is coded A through E.

The test data consists of 20 observations of 160 variables.

#Analysis
##Data Cleaning
I began the analysis by cleaning up the training data set.  A number of variables consisted largely of missing or NA data.  So I began by removing variables that contained NA values.  Also removed the `classe` variable, along with several other variables that were not likely to be predictive such as the record number, user name, timestamp, and the window data.

I then converted the remaining variables to numeric as many of them had been incorrectly read in as factor or logical variables.  I then re-added the `classe` variable to the data set.

This left me with a training data set consisting of 19622 observations of 86 variables.


```r
setwd("C:/Users/Derek/Documents/My Dropbox/Projects/machine_learning")
train <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")

filteredTrain <- train[, !unlist(lapply(train, function(x) any(is.na(x))))]
filteredTrain <- filteredTrain[, -c(1:7,93)]
filteredTrain <- as.data.frame(lapply(filteredTrain, as.numeric))
filteredTrain <- cbind(classe = train$classe, filteredTrain)
dim(filteredTrain)
```

```
## [1] 19622    86
```

##Modeling
I decided to use the `gbm` package.  This is a stochastic gradient boosted model that is quite robust for both classification and regression.  Because it is fundamentally a decision tree based model, it is also quite good at dealing with a large number of variables (many of which may be non-predictive) without having to engage in PCA or other variable selection techniques.

I used a 10-fold cross validation approach to optimize the model.  Additionally, I used a custom parameter tuning grid.  I checked the model at a 3, 5, and 7 level interaction depth, 50 through 250 trees, and both .01 and .1 shrinkage levels (essentially a "slow" and "fast" learner).


```r
fitControl <- trainControl(method = "cv",
                           number = 10)

gbmGrid <-  expand.grid(interaction.depth = c(3, 5, 7),
                        n.trees = (1:5)*50,
                        shrinkage = c(0.01, 0.1),
                        n.minobsinnode = 10)
```




```r
set.seed(42)
gbmFit <- train(classe ~ ., data = filteredTrain,
                method = "gbm",
                trControl = fitControl,
                tuneGrid = gbmGrid,
                verbose = FALSE)
```

##Final model and error estimates
Optimal accuracy was achieved with 250 trees, an interaction depth of 7 and a shrinkage of 0.1.  This approach achieved accuracy of 0.9937312 with a SD of 0.001273787 on the training data.  Additionally, the Kappa statistic is greater than 0.99.  This is a comparison of the observed accuracy to the expected accuracy (of random chance).  A Kappa statistic greater than .75 is typically considered an excellent model.  A Kappa of greater than 0.99 suggests a near perfect model.

All of this suggests that we're likely to see better than .99 out-of-sample accuracy given that the 10-fold cross-validation is still slightly biased towards the training data and gbm models, like many tree based models, can occassionally overfit the training data.


```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    85 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 17659, 17661, 17661, 17659, 17661, 17660, ... 
## 
## Resampling results across tuning parameters:
## 
##   shrinkage  interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   0.01       3                   50      0.6977359  0.6178426  0.013308214
##   0.01       3                  100      0.7462029  0.6782828  0.008991180
##   0.01       3                  150      0.7864635  0.7292450  0.008675626
##   0.01       3                  200      0.8187222  0.7702583  0.008717274
##   0.01       3                  250      0.8402289  0.7974158  0.008598124
##   0.01       5                   50      0.7693403  0.7086241  0.005669367
##   0.01       5                  100      0.8221886  0.7748153  0.008757988
##   0.01       5                  150      0.8562320  0.8178357  0.007985834
##   0.01       5                  200      0.8777390  0.8450504  0.008830412
##   0.01       5                  250      0.8954237  0.8675240  0.007532671
##   0.01       7                   50      0.8038426  0.7518728  0.005076420
##   0.01       7                  100      0.8536848  0.8146174  0.009200040
##   0.01       7                  150      0.8870653  0.8569209  0.007131575
##   0.01       7                  200      0.9081644  0.8836841  0.004629913
##   0.01       7                  250      0.9256961  0.9058938  0.005369372
##   0.10       3                   50      0.8974106  0.8701265  0.004539249
##   0.10       3                  100      0.9446546  0.9299737  0.005208169
##   0.10       3                  150      0.9633581  0.9536444  0.004484716
##   0.10       3                  200      0.9731936  0.9660864  0.004409988
##   0.10       3                  250      0.9799715  0.9746637  0.003294050
##   0.10       5                   50      0.9400169  0.9240731  0.004019591
##   0.10       5                  100      0.9714614  0.9638973  0.004988676
##   0.10       5                  150      0.9830292  0.9785317  0.002610798
##   0.10       5                  200      0.9889919  0.9860758  0.001983461
##   0.10       5                  250      0.9919985  0.9898790  0.001968325
##   0.10       7                   50      0.9606059  0.9501489  0.004044796
##   0.10       7                  100      0.9822141  0.9775011  0.003466345
##   0.10       7                  150      0.9896542  0.9869135  0.002096280
##   0.10       7                  200      0.9927120  0.9907813  0.001361045
##   0.10       7                  250      0.9937312  0.9920706  0.001273787
##   Kappa SD   
##   0.016994253
##   0.011555418
##   0.011211848
##   0.011117680
##   0.010960328
##   0.007316773
##   0.011248956
##   0.010235066
##   0.011297063
##   0.009633845
##   0.006542647
##   0.011821900
##   0.009121241
##   0.005933165
##   0.006836364
##   0.005746459
##   0.006593864
##   0.005676598
##   0.005580605
##   0.004167620
##   0.005102061
##   0.006308868
##   0.003302018
##   0.002508113
##   0.002489026
##   0.005123751
##   0.004384578
##   0.002650753
##   0.001721591
##   0.001610912
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 250,
##  interaction.depth = 7, shrinkage = 0.1 and n.minobsinnode = 10.
```

```
## A gradient boosted model with multinomial loss function.
## 250 iterations were performed.
## There were 85 predictors of which 52 had non-zero influence.
```

![](project_writeup_files/figure-html/unnamed-chunk-5-1.png) 

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are cell counts per resample)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 5571   20    0    0    0
##          B    8 3765   16    1    2
##          C    1   12 3394   29    3
##          D    0    0   12 3181   14
##          E    0    0    0    5 3588
```

Additionally, we can look at the relative variable importance of the final model.  

```r
varImp(gbmFit)
```

```
## gbm variable importance
## 
##   only 20 most important variables shown (out of 85)
## 
##                   Overall
## roll_belt         100.000
## yaw_belt           53.814
## pitch_forearm      48.813
## magnet_dumbbell_z  37.479
## magnet_dumbbell_y  29.847
## roll_forearm       26.492
## pitch_belt         25.355
## magnet_belt_z      19.153
## roll_dumbbell      14.797
## gyros_belt_z       14.744
## yaw_arm            13.149
## accel_dumbbell_y   13.048
## magnet_forearm_z   12.627
## accel_forearm_z    12.070
## accel_forearm_x    11.203
## accel_dumbbell_z   10.087
## magnet_dumbbell_x  10.086
## accel_dumbbell_x    8.229
## magnet_forearm_x    7.325
## magnet_belt_x       7.320
```




#Test data
Some minimal cleaning was required to prepare the test data set.  Essentially, all of the variables had to be converted to numeric values and NA values were replaced with 0.

Once that was done, as expected, the `gbm` model exhibited high out-of-sample accuracy, correctly predicting all 20 test observations.


```r
numericTest <- as.data.frame(lapply(test, as.numeric))
numericTest[is.na(numericTest)] <- 0
gbmPred <- predict(gbmFit, numericTest)
gbmPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
