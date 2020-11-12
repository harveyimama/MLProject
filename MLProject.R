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
