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
