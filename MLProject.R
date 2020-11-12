#Author :  Harvey imama
#NYC data science academy ML project


###############################
# Data Analysis and missingness
###############################
houses = read.csv('Downloads/ML_Project/train.csv')

sapply(houses, sd) 


library(Hmisc)
set.seed(0)
houses$GarageYrBlt = Hmisc::impute(houses$GarageYrBlt, "random")
houses$MasVnrArea = Hmisc::impute(houses$MasVnrArea, "random")
houses$LotFrontage = Hmisc::impute(houses$LotFrontage, "random")

features <- setdiff(colnames(houses), c("Id", "SalePrice"))
for (f in features) {
  if (any(is.na(houses[[f]]))) 
    if (is.character(houses[[f]])){ 
      houses[[f]][is.na(houses[[f]])] <- "None"
    }
}

bound = nrow(houses)*.7
houses.train =  houses[1:bound,]
houses.test =  houses[bound:nrow(houses), ]


###############################
# Multiple linear regression and 
# feature selection
###############################

model.all = lm(SalePrice ~  .
               , data = houses.train )

summary(model.all)
plot(model.all)
library(car)
influencePlot(model.all)
vif(model.all)
avPlots(model.all)

summary(model.all)

model.selected = lm(SalePrice ~ MSZoning+LotArea+LandSlope+Neighborhood
                    + BldgType +OverallQual 
                    + ExterQual + BsmtFinSF1 + X1stFlrSF + X2ndFlrSF + FullBath
                    + BedroomAbvGr + KitchenAbvGr + KitchenQual + TotRmsAbvGrd
                    + Functional + Fireplaces + GarageType + GarageArea + GarageQual
                    + WoodDeckSF + ScreenPorch + MoSold + SaleType,   data = houses.train 
)


summary(model.selected)
plot(model.selected)
influencePlot(model.selected)
vif(model.selected)
avPlots(model.selected)


model.selected.second = lm(SalePrice ~ MSZoning+LotArea +LandSlope
                           + Neighborhood  + BldgType + OverallQual 
                           + ExterQual + BsmtFinSF1 +X1stFlrSF +X2ndFlrSF + FullBath
                           + BedroomAbvGr + KitchenAbvGr+ KitchenQual +TotRmsAbvGrd
                           + Functional + GarageType + GarageArea + WoodDeckSF + ScreenPorch
                           + MoSold + SaleType, data = houses.train 
)

summary(model.selected.second)
plot(model.selected.second)
influencePlot(model.selected.second)
vif(model.selected.second)
avPlots(model.selected.second)


AIC(model.selected,model.all,model.selected.second)
BIC(model.selected,model.all,model.selected.second)

#best model plot
lm.pred = predict(model.selected.second,houses.test)

lm.pred = data.frame(predicted = lm.pred, actual =houses.test$SalePrice )

plot(lm.pred) 
abline(0, 1)

###############################
# Regularized regression
# Ridge and Lasso Regression using cross validation
###############################

library(ISLR)
x = model.matrix(SalePrice ~ .,  houses)[, -1]
y = houses$SalePrice


train = sample(1:nrow(x), 7*nrow(x)/10)
test = (-train)
y.test = y[test]

library(glmnet)

ridge.house.train = glmnet(x[train, ], y[train], alpha = 0, lambda = grid)
grid = 10^seq(5, -2, length = 100)
cv.ridge.house = cv.glmnet(x[train,], y[train],
                           lambda = grid, alpha = 0, nfolds = 10)

ridge.predict = predict(ridge.house.train, s = cv.ridge.house$lambda.min, newx = x[test, ])
mean((ridge.predict - y.test)^2)

ridge.pred = data.frame(predicted = ridge.predict, actual =y.test )

plot(ridge.pred)
abline(0, 1)


lasso.house.train = glmnet(x[train, ], y[train], alpha = 1, lambda = grid)

cv.lasso.house = cv.glmnet(x[train,], y[train],
                           lambda = grid, alpha = 0, nfolds = 10)

lasso.predict = predict(lasso.house.train, s = cv.lasso.house$lambda.min, newx = x[test, ])
mean((lasso.predict - y.test)^2)

lasso.pred = data.frame(predicted = lasso.predict, actual =y.test )

plot(lasso.pred)
abline(0, 1)

###############################
# Tree Based models
# bagging ,random forests and boosting
###############################


library(tree)
library(MASS)
tree.house = tree(SalePrice ~ ., houses, subset = train)
summary(tree.house)

plot(tree.house)
text(tree.house, pretty = 0)

cv.tree.house = cv.tree(tree.house)
par(mfrow = c(1, 2))
plot(cv.tree.house$size, cv.tree.house$dev, type = "b",
     xlab = "Terminal Nodes", ylab = "RSS")
plot(cv.tree.house$k, cv.tree.house$dev, type  = "b",
     xlab = "Alpha", ylab = "RSS")

prune.house = prune.tree(tree.house, best = 4)
par(mfrow = c(1, 1))
plot(prune.house)
text(prune.house, pretty = 0)

#Calculating and assessing the MSE of the test data on the overall tree.
tree.predict = predict(tree.house, newdata = houses[-train, ])
house.test = houses[-train, "SalePrice"]
mean((tree.predict - house.test)^2)

tree.pred = data.frame(predicted = tree.predict, actual =house.test )
plot(tree.pred)
abline(0, 1)


tree.prune.predict = predict(prune.house, newdata = houses[-train, ])
mean((tree.prune.predict - house.test)^2)

tree.prune.pred = data.frame(predicted = tree.prune.predict, actual =house.test )
plot(tree.prune.pred)
abline(0, 1)

library(randomForest)
oob.err = numeric(80)
for (mtry in 1:80) {
  fit = randomForest(SalePrice ~ ., data = houses[train, ], mtry = mtry)
  oob.err[mtry] = fit$mse[500]
}

#Visualizing the OOB error.
plot(1:80, oob.err, pch = 16, type = "b",
     xlab = "Variables Considered at Each Split",
     ylab = "OOB Mean Squared Error",
     main = "Random Forest OOB Error Rates\nby # of Variables")

rf.house = randomForest(SalePrice ~ ., data = houses[train, ], mtry = 15)
importance(rf.house)
varImpPlot(rf.house)
rf.predicted = predict(rf.house,houses[test, ])

rf.pred = data.frame(predicted = rf.predicted, actual =house.test )
plot(rf.pred)
abline(0, 1)

rf.house.bagged = randomForest(SalePrice ~ ., data = houses[train, ], mtry = 80)
importance(rf.house)
varImpPlot(rf.house)
rf.predicted.bagged = predict(rf.house.bagged,houses[test, ])

rf.pred.bagged = data.frame(predicted = rf.predicted.bagged, actual =house.test )
plot(rf.pred.bagged)
abline(0, 1)

library(gbm)
houses = read.csv('Downloads/ML_Project/train.csv',stringsAsFactors=TRUE)
real.test = read.csv('Downloads/ML_Project/test.csv',stringsAsFactors=TRUE)
boost.house = gbm(SalePrice ~ ., data = houses[train, ],
                  distribution = "gaussian",
                  n.trees = 10000,
                  interaction.depth = 4,
                  shrinkage = 0.001)

n.trees = seq(from = 100, to = 10000, by = 100)
boost.house.prediction = predict(boost.house,
                                 newdata = houses[test, ],
                                 n.trees = n.trees,
                                 type = "response")
boost.house.prediction = round(boost.house.prediction)

berr2 = with(houses[-train, ], apply((boost.house.prediction - SalePrice)^2, 2, mean))
plot(n.trees, berr2, pch = 16,
     ylab = "Mean Squared Error",
     xlab = "# Trees",
     main = "Boosting Test Error")

boost.house.prediction = predict(boost.house,
                                 newdata = houses[test, ],
                                 n.trees = 10000,
                                 type = "response")

boost.pred.boosted = data.frame(predicted = boost.house.prediction, actual =house.test )
plot(boost.pred.boosted)
abline(0, 1)


###############################
# Final Result
###############################

predicted.final = data.frame(predicted = boost.house.prediction, actual =real.test )

predicted.final

