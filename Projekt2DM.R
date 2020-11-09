df_train<-read.csv("C:/Users/user/Desktop/DM/PROJEKT2/train_projekt2.csv", header = TRUE)

df_test <- read.csv("C:/Users/user/Desktop/DM/PROJEKT2/test_projekt2.csv", header = TRUE)

plot(df_train$V1, df_train$V2, col=as.numeric(df_train$Y)+1)


if(!require(mlbench)){
  install.packages("mlbench")
  library(mlbench)
}

if(!require(infotheo)){
  install.packages("infotheo")
  library(infotheo)
}

if(!require(ROCR)){
  install.packages("ROCR")
  library(ROCR)
}

if(!require(MASS)){
  install.packages("MASS")
  library(MASS)
}
if(!require(rpart)){
  install.packages("rpart")
  library(rpart)
}
if(!require(adabag)){
  install.packages("adabag")
  library(adabag)
}
if(!require(ggplot2)){
  install.packages("ggplot2")
  library(ggplot2)
}

if(!require(Boruta)){
  install.packages("Boruta")
  library(Boruta)
}

if(!require(randomForest)){
  install.packages("randomForest")
  library(randomForest)
}

CMIMselection<-function(X,y, kmax){
  
  stopifnot(all(!is.na(X)) & nrow(X)==length(y) & ncol(X)>1 & kmax<=ncol(X) & kmax==floor(kmax))
  N<-ncol(X)
  S<-rep(0,N)
  score<-rep(0,N)
  nu<-rep(0,N)
  ps<-rep(0,N)
  
  for(i in (1:N))
  {
    score[i]<-mutinformation(y, X[,i])
  }
  
  for(k in (1:kmax))
  {
    ps[k]<-0
    for(j in (1:N))
    {
      while((score[j]>ps[k]) & (S[j]<k-1) )
      {
        S[j]<-S[j]+1
        score[j]<-min(score[j], condinformation(y, X[,j], X[,nu[S[j]]]))
      }
      if(score[j]>ps[k])
      {
        ps[k]<-score[j]
        nu[k]<-j
      }
      
    }
    
  }
  return(list(S = nu[1:kmax], score = ps[1:kmax]))
  
} 



#``````podzielenie zbioru treningowego na treningowy i testowy, na treningowym selekcja


train_rows <- sample(1:2000, 0.8*2000 )
train <- df_train[train_rows,]
test <- df_train[-train_rows,]






library(ROCR)
# ``````````````` roc dla logistycznego z cmim 9 kolumn`




auc_log<-c()
for(i in (1:20)){
  
  CMIM_sel <- CMIMselection(train[,-501], train[,501],i)
  train_sel <- train[,c(CMIM_sel$S, 501)]
  test_sel <- test[,c(CMIM_sel$S, 501)]
  
  model_log_sel <- glm(Y~., data=train_sel, family = binomial)
  log_pred_sel <- predict(model_log_sel, newdata = test_sel, type = "response")
  
  pred <- prediction(log_pred_sel, test_sel$Y)
  auc.perf = performance(pred, measure = "auc")
  auc_log[i] <- auc.perf@y.values
}

#-----------lda dla cmimu
auc_lda<-c()
for(i in (1:20)){
  
  CMIM_sel <- CMIMselection(train[,-501], train[,501],i)
  train_sel <- train[,c(CMIM_sel$S, 501)]
  test_sel <- test[,c(CMIM_sel$S, 501)]
  
  model_lda_sel <- lda(Y~., data=train_sel)
  lda_pred_sel <- predict(model_lda_sel, newdata = test_sel)
  
  pred <- prediction(lda_pred_sel$posterior[,2], test_sel$Y)
  auc.perf = performance(pred, measure = "auc")
  auc_lda[i] <- auc.perf@y.values
}


# dla drzewa decyzyjnego
auc_dt<-c()
for(i in (1:20)){
  
  CMIM_sel <- CMIMselection(train[,-501], train[,501],i)
  train_sel <- train[,c(CMIM_sel$S, 501)]
  test_sel <- test[,c(CMIM_sel$S, 501)]
  
  tree_sel<- rpart(as.factor(Y)~., data=train_sel,
                   cp=0.01, minsplit=5)
  tree_pred_sel <- predict(tree_sel, newdata = test_sel)
  
  pred <- prediction(tree_pred_sel[,2], test_sel$Y)
  auc.perf = performance(pred, measure = "auc")
  auc_dt[i] <- auc.perf@y.values
}




CMIM_tree <- CMIMselection(train[,-501], train[,501], 14)
train_sel_tree <- train[,c(CMIM_tree$S, 501)]
test_sel_tree <- test[,c(CMIM_tree$S, 501)]

tree_sel<- rpart(as.factor(Y)~., data=train_sel_tree,
                 cp=0.01, minsplit=5)
tree_pred_sel <- predict(tree_sel, newdata = test_sel_tree)

pred <- prediction(tree_pred_sel[,2], test_sel_tree$Y)
auc.perf = performance(pred, measure = "auc")
auc <- auc.perf@y.values

perf <- performance(pred, "tpr", "fpr")
plot(perf)

plotcp(tree_sel)
tree_sel$cptable[which.min(tree_sel$cptable[,"xerror"]), "CP"]

tree.2 <- prune.rpart(tree_sel, cp=0.012)


tree_pred_sel <- predict(tree.2, newdata = test_sel_tree)

pred <- prediction(tree_pred_sel[,2], test_sel_tree$Y)
auc.perf = performance(pred, measure = "auc")
auc <- auc.perf@y.values


perf <- performance(pred, "tpr", "fpr") 
plot(perf)




# bagging z pakietu adabag
library(adabag)
test$Y<-as.factor(test$Y)
train$Y<-as.factor(train$Y)
auc_b<-c()
for(i in (1:20)){
  
  CMIM_sel <- CMIMselection(train[,-501], train[,501],i)
  train_sel <- train[,c(CMIM_sel$S, 501)]
  test_sel <- test[,c(CMIM_sel$S, 501)]
  
  bag <- bagging(Y~., data = train_sel)
  pred_bag <- predict(bag, newdata =test_sel)
  
  pred <- prediction(pred_bag$prob[,2], test_sel$Y)
  auc.perf = performance(pred, measure = "auc")
  auc_b[i] <- auc.perf@y.values
}


# adaboost----------------------------
auc_a<-c()
for(i in (8:20)){
  
  CMIM_sel <- CMIMselection(train[,-501], train[,501],i)
  train_sel <- train[,c(CMIM_sel$S, 501)]
  test_sel <- test[,c(CMIM_sel$S, 501)]
  
  ada <- boosting(Y~., data = train_sel)
  pred_ada <- predict(ada, newdata =test_sel)
  
  pred <- prediction(pred_ada$prob[,2], test_sel$Y)
  auc.perf = performance(pred, measure = "auc")
  auc_a[i] <- auc.perf@y.values
}


aucax<-c()
for(i in (1:10)){
  

CMIM_tree <- CMIMselection(train[,-501], train[,501], 10)
train_sel_tree <- train[,c(CMIM_tree$S, 501)]
test_sel_tree <- test[,c(CMIM_tree$S, 501)]

ada <- boosting(Y~., data=train_sel_tree)
pred.ada <- predict(ada, newdata = test_sel_tree)
pred <- prediction(pred.ada$prob[,2], test_sel_tree$Y)
auc.perf = performance(pred, measure = "auc")
aucax[i] <- auc.perf@y.values
}
mean(unlist(aucax))

unlist(aucax)
#--------------random forest
df_train<-read.csv("C:/Users/user/Desktop/DM/PROJEKT2/train_projekt2.csv", header = TRUE)
df_test <- read.csv("C:/Users/user/Desktop/DM/PROJEKT2/test_projekt2.csv", header = TRUE)
train_rows <- sample(1:2000, 0.8*2000 )
train <- df_train[train_rows,]
test <- df_train[-train_rows,]
auc_r<-c()
for(i in (5:20)){
  
  CMIM_sel <- CMIMselection(train[,-501], train[,501],i)
  train_sel <- train[,c(CMIM_sel$S, 501)]
  test_sel <- test[,c(CMIM_sel$S, 501)]
  
  forest <- randomForest(Y~., data = train_sel)
  pred_forest <- predict(forest, newdata =test_sel)
  
  pred <- prediction(pred_forest, as.factor(test_sel$Y))
  auc.perf = performance(pred, measure = "auc")
  auc_r[i-4] <- auc.perf@y.values
}



aucarx<-c()
for(i in (1:10)){
  
  
  CMIM_tree <- CMIMselection(train[,-501], train[,501], 16)
  train_sel_tree <- train[,c(CMIM_tree$S, 501)]
  test_sel_tree <- test[,c(CMIM_tree$S, 501)]
  
  
  forest <- randomForest(Y~., data = train_sel)
  pred_forest <- predict(forest, newdata =test_sel)
  
  pred <- prediction(pred_forest, as.factor(test_sel$Y))
  auc.perf = performance(pred, measure = "auc")
  
  aucarx[i] <- auc.perf@y.values
}
mean(unlist(aucarx))






# boruta

library(Boruta)

boruta.fs <- Boruta(Y~., data = train, doTrace=2)
conf<- boruta.fs$finalDecision[boruta.fs$finalDecision=="Confirmed"]


train_sel_boruta <-train[,c(29, 49, 65, 106, 129, 154, 242, 282, 319, 337,339,379,434,443,452,454,456,473,476,494, 501)]
test_sel_boruta <-test[,c(29, 49, 65, 106, 129, 154, 242, 282, 319, 337,339,379,434,443,452,454,456,473,476,494, 501)]

model_log_sel_boruta <- glm(Y~., data=train_sel_boruta, family = binomial)
log_pred_sel <- predict(model_log_sel_boruta, newdata = test_sel_boruta, type = "response")
log_pred_sel
pred <- prediction(log_pred_sel, test_sel_boruta$Y)
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values




# bagging z adabag
train_sel_tree$Y<-as.factor(train_sel_tree$Y)

train_sel_boruta$Y<-as.factor(train_sel_boruta$Y)
test_sel_boruta$Y<-as.factor(test_sel_boruta$Y)
library(adabag)

aucborx<-c()
for(i in(1:10)){
  

bag_boruta <- bagging(Y~., data = train_sel_boruta)
pred_bag_boruta <- predict(bag_boruta, newdata =test_sel_boruta)

pred <- prediction(pred_bag_boruta$prob[,2] , test_sel_boruta$Y)
auc.perf = performance(pred, measure = "auc")
aucborx[i] <- auc.perf@y.values

}
mean(unlist(aucborx))

# adaboost
aucaborx<-c()
for(i in (1:10)){
  
  

  ada <- boosting(Y~., data=train_sel_boruta)
  pred_ada <- predict(ada, newdata = test_sel_boruta)
  pred <- prediction(pred_ada$prob[,2], test_sel_boruta$Y)
  auc.perf = performance(pred, measure = "auc")
  aucaborx[i] <- auc.perf@y.values
}
mean(unlist(aucaborx))

#random forest
aucrfborx<-c()
for(i in (1:10)){
  
  

  forest <- randomForest(Y~., data = train_sel_boruta)
  pred_forest <- predict(forest, newdata =test_sel_boruta)
  
  pred <- prediction(pred_forest, as.factor(test_sel_boruta$Y))
  auc.perf = performance(pred, measure = "auc")
  
  aucrfborx[i] <- auc.perf@y.values
}


mean(unlist(aucrfborx))
# ostateczne wyniki
df_test<-df_test[,c(29, 49, 65, 106, 129, 154, 242, 282, 319, 337,339,379,434,443,452,454,456,473,476,494)]
forest_top <- randomForest(Y~., data = train_sel_boruta)
pred_forest <- predict(forest_top, newdata =df_test)



x<-as.vector(pred_forest)

write.table(x, file = "MPA.txt")

            