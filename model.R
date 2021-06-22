rm(list = ls())

library(tidyverse)
library(doParallel)
library(foreach)

library(pracma)
library(signal)
library(seewave)
library(fBasics)
library(changepoint)

library(data.table)
library(mltools)
library(e1071)
library(caret)
library(randomForest)
library(xgboost)
library(class)
library(keras)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 작업 디렉토리 설정
setwd("/Users/kwondoyun/Downloads/mit")

# 파일명들 불러오기
d <- getwd()
fls <- dir(d, recursive = T)
fls <- fls[1:18]

# ------------------------------------------------------------------------------
# File에서 IBP만 추출
count <- 1
system.time(
  for(flist in fls){
    path <- file.path(str_c(d, "/", flist))
    data <- read.csv(path)
    signal <- data$signal[3:length(data$signal)]
    split_signal <- strsplit(signal, ",")
  
    ibp <- as.numeric(unlist(lapply(split_signal, function(d){
      return(d[2])
    })))
  
    assign(flist, ibp)
    
    print(count)
    count <- count + 1
  }
)

# ------------------------------------------------------------------------------
# Data set 만들기
SRATE<-250
MINUTES_AHEAD<-1
Data_set<-list() 
for (file in fls){
  IBP <- as.numeric(get(file))
  i <- 1
  IBP_data<-data.frame()
  while (i < length(IBP) - SRATE*(1+1+MINUTES_AHEAD)*60){
    segx <- IBP[i:(i+SRATE*1*60-1)]
    segy <- IBP[(i+SRATE*(1+MINUTES_AHEAD)*60):(i+SRATE*(1+1+MINUTES_AHEAD)*60-1)]
    segxd <- IBP[i:(i+SRATE*(1+MINUTES_AHEAD)*60-1)]
    if(is.na(mean(segx)) |
       is.na(mean(segy)) |
       max(segx)>200 | min(segx)<20 |
       max(segy)>200 | max(segy)<20 |
       max(segx) - min(segx) < 30 |
       max(segy) - min(segy) < 30|(min(segxd,na.rm=T) <= 50)){
    }
    else{ #나머지의 경우
      # segy <- ma(segy, 2*SRATE)
      event <- ifelse(min(segy,na.rm=T) <= 50, 1, 0)
      print(event)
      IBP_data<- rbind(IBP_data, cbind(t(segx), event))
    }
    
    i <- i+1*60*SRATE
  }
  Data_set[[file]]<- IBP_data
}

# 18 Files 합치기
ibp_data <- data.frame()
for(data in Data_set){
  ibp_data <- rbind(ibp_data, data)
}

# 저장
# saveRDS(ibp_data, "ibp_data.rds")
ibp_data <- readRDS("ibp_data.rds")
ibp_data$event <- as.factor(ibp_data$event)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# statistics
table(ibp_data$event)
ibp_data[is.na(ibp_data)] <- 0
normal <- subset(ibp_data, ibp_data$event == 0)
hypo <- subset(ibp_data, ibp_data$event == 1)

total <- ibp_data[,1:15000]
normal <- normal[,1:15000]
hypo <- hypo[,1:15000]

# mean
mean(apply(total, 1, mean))
mean(apply(normal, 1, mean))
mean(apply(hypo, 1, mean))

# min
mean(apply(total, 1, min))
mean(apply(normal, 1, min))
mean(apply(hypo, 1, min))

# max
mean(apply(total, 1, max))
mean(apply(normal, 1, max))
mean(apply(hypo, 1, max))

# sd
mean(apply(total, 1, sd))
mean(apply(normal, 1, sd))
mean(apply(hypo, 1, sd))

# skewness
mean(apply(total, 1, skewness))
mean(apply(normal, 1, skewness))
mean(apply(hypo, 1, skewness))

# kurtosis
mean(apply(total, 1, kurtosis))
mean(apply(normal, 1, kurtosis))
mean(apply(hypo, 1, kurtosis))

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 특징 추출
temp <- as.data.frame(t(subset(ibp_data, ibp_data$event==1)[1,]))
plot(temp[1:15000,], type = "l")
abline(h = 50, col = "red")

ibp_ext <- as.data.frame(t(ibp_data[,1:15000]))
str(ibp_ext)

# ------------------------------------------------------------------------------
# 통계 특징 추출
# skewness (왜도) 함수 만들기
skewness <- function(x){
  (sum((x - mean(x))^3)/length(x))/((sum((x - mean(x))^2)/length(x)))^(3/2)
}

# rrs 함수 만들기
rrs <- function(x) rms(x)*(length(x))^0.5

harmonic_mean <- function(x){
  1/mean(1/x)
}
# ------------------------------------------------------------------------------
# 특징 추출
smry_rslt <- data.frame()
for (name in colnames(ibp_ext)){
  temp <- ibp_ext %>% summarize_at(.vars = c(name),
                                   .funs = c(mean, min, max, sd, skewness, rms, rrs, IQR, e1071::kurtosis, harmonic_mean))
  smry_rslt <- rbind(smry_rslt, temp)
}


# ------------------------------------------------------------------------------
# 피크 특징 추출
Peak_rslt <- data.frame()
for(sub in ibp_ext){

  p <- findpeaks(-sub)
  p <- subset(p, p[,1] > -80)
  Peak_rslt <- rbind(Peak_rslt, data.frame(f_n=ifelse(!is.null(p), dim(p)[1], 0),
                                           p_interval=ifelse(!is.null(p), ifelse(dim(p)[1]>2, mean(diff(p[,2])), 0), 0),
                                           p_interval_std=ifelse(!is.null(p), ifelse(dim(p)[1]>2, std(diff(p[,2])), 0), 0),
                                           p_mean=-ifelse(!is.null(p), ifelse(is.na(mean(p[,1])), 0, mean(p[,1])), 0),
                                           p_min=-ifelse(!is.null(p), ifelse(max(p[,1])==-Inf, 0, max(p[,1])), 0),
                                           p_max=-ifelse(!is.null(p), ifelse(min(p[,1])==Inf, 0, min(p[,1])), 0),
                                           p_std=ifelse(!is.null(p), ifelse(is.null(std(p[,1])), 0, std(p[,1])), 0)))
}

Peak_rslt[is.na(Peak_rslt)] <- 0

# ------------------------------------------------------------------------------
# 파고율 추출
cf_df <- data.frame()
for (sub in ibp_ext){

  sub_ibp <- -sub
  cf_sub <- crest(sub_ibp, f=250, plot = F)
  
  cf_df <- rbind(cf_df, data.frame(cf_ibp=cf_sub$C))
}

# ------------------------------------------------------------------------------
# 변화 특징 추출
rslt <- sapply(ibp_ext, cpt.mean)
rslt2 <- sapply(ibp_ext, cpt.var)
rslt3 <- sapply(ibp_ext, cpt.meanvar)

cpt_mean <- c()
for(cptm in rslt){
  cpt_mean <- c(cpt_mean, cpts(cptm))
}

cpt_var <- c()
for(cptv in rslt2){
  cpt_var <- c(cpt_var, cpts(cptv))
}

cpt_mean_var <- c()
for(cptmv in rslt){
  cpt_mean_var <- c(cpt_mean_var, cpts(cptmv))
}

# ------------------------------------------------------------------------------
# 퓨리에 변환
# 조화평균 함수 만들기
harmonic_mean <- function(x){
  1/mean(1/x)
}

# 변이계수 함수 만들기
coefficient_var <- function(x){
  100*sd(x)/mean(x)
}

fftsmry_df <- data.frame()
for(sub in ibp_ext){
  fft_sub <- fft(sub)
  revalue <- Re(fft_sub)
  imvalue <- Im(fft_sub)
  temp <- data.frame(revalue, imvalue)
  
  fft_temp <- temp %>% 
    summarize_at(.vars = c("revalue", "imvalue"),
                 .funs = c(mean, min, max, sd, skewness, rms, rrs, IQR, e1071::kurtosis, harmonic_mean, coefficient_var, weighted.mean))
  
  fftsmry_df <- rbind(fftsmry_df, data.frame(fft_temp))
}

ibp_feature_df <- cbind(smry_rslt, cf_df, fftsmry_df, Peak_rslt)
ibp_feature_df$cpt_mean <- cpt_mean
ibp_feature_df$cpt_var <- cpt_var
ibp_feature_df$cpt_mean_var <- cpt_mean_var 
ibp_feature_df$event <- as.factor(ibp_data$event)
#saveRDS(ibp_feature_df, "ibp_modeling_data2.rds")
#load("ibp_modeling_data2.rds")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# PCA
pcadata <- ibp_feature_df %>% select(-event, -imvalue_fn10)
pca <- prcomp(pcadata, scale = TRUE)

# PC1에 대해 강한 양의 부하량 : fn10, fn3, fn7 등  / 음의 부하량 : p_std, p_max, p_mean
# PC2에 대해 강한 양의 부하량 : revalue_fn9, imvalue_fn2 등 / 음의 부하량 : imvaluefn3, fn4 등
screeplot(pca, main = "", col = "green", type = "lines", pch = 1, npcs = length(pca$sdev))
biplot(pca)

summary(pca)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Modeling
# Data set split
set.seed(234)
inTrain = createDataPartition(1:nrow(ibp_feature_df), p=0.8, list=FALSE)

train = ibp_feature_df[inTrain,]
test = ibp_feature_df[-inTrain, ]
table(train$event)
table(test$event)
nrow(train)
nrow(test)

train_up = upSample(subset(train, select=-event), train$event)
train_down = downSample(subset(train, select=-event), train$event)

train_x = train %>% select(-event)
train_y = train$event

train_x_up = train_up %>% select(-Class)
train_y_up = train_up$Class
nrow(train_x_up)
table(train_y_up)

train_x_down = train_down %>% select(-Class)
train_y_down = train_down$Class
nrow(train_x_down)
table(train_y_down)

test_x = test %>% select(-event)
test_y = test$event
# ------------------------------------------------------------------------------

pca_data2 = pcadata
pca_data2$event = as.factor(ibp_feature_df$event)

train_pca = pca_data2[inTrain,]
test_pca = pca_data2[-inTrain,]

train_x_pca = train_pca %>% select(-event)
train_x_pca <- as.matrix(train_x_pca) %*% pca$rotation
train_y_pca = train_pca$event

test_x_pca = test_pca %>% select(-event)
test_x_pca <- as.matrix(test_x_pca) %*% pca$rotation
test_y_pca = test_pca$event

train_pca_rf = as.data.frame(train_x_pca)
train_pca_rf$event <- as.factor(train_y_pca)
# ------------------------------------------------------------------------------
confusion_matrix = function(result){
  accuracy <- c()
  precision <- c()
  recall <- c()
  accuracy_normal <- c()
  precision_normal <- c()
  recall_normal <- c()

  TP <- result[1]
  FN <- result[2]
  FP <- result[3]
  TN <- result[4]
    
  accuracy <- c(accuracy, (TP+TN)/(TP+FP+FN+TN))
  precision <- c(precision, TP/(TP+FP))
  recall <- c(recall, TP/(TP+FN))
    
  accuracy_normal <- c(accuracy_normal, (TP+TN)/(TP+FP+FN+TN))
  precision_normal <- c(precision_normal, TN/(TN+FN))
  recall_normal <- c(recall_normal, TN/(TN+FP))

  
  result <- data.frame(normal = c(accuracy, precision, recall),
                       event = c(accuracy_normal, precision_normal, recall_normal))
  
  rownames(result) <- c("accuracy", "precision", "recall")
  return(result)
}

# ------------------------------------------------------------------------------
# RF
rf.fit_raw = randomForest(event ~ ., data=train, mtry=floor(sqrt(length(train)-1)), ntree=500, importance=T)
rf.fit_up = randomForest(Class ~ ., data=train_up, mtry=floor(sqrt(length(train_up)-1)), ntree=500, importance=T)
rf.fit_down = randomForest(Class ~ ., data=train_down, mtry=floor(sqrt(length(train_down)-1)), ntree=500, importance=T)
rf.fit_pca = randomForest(event ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12, data=train_pca_rf, mtry=floor(sqrt(length(train_pca_rf)-1)), ntree=500, importance=T)

y_predrf_raw = predict(rf.fit_raw, test_x)
y_predrf_up = predict(rf.fit_up, test_x)
y_predrf_down = predict(rf.fit_down, test_x)
y_predrf_pca = predict(rf.fit_pca, test_x_pca)

conf_rf_raw <- confusionMatrix(y_predrf_raw, test_y)
conf_rf_up <- confusionMatrix(y_predrf_up, test_y)
conf_rf_down <- confusionMatrix(y_predrf_down, test_y)
conf_rf_pca <- confusionMatrix(y_predrf_pca, test_y_pca)

confusion_matrix(conf_rf_raw$table)
confusion_matrix(conf_rf_up$table)
confusion_matrix(conf_rf_down$table)
confusion_matrix(conf_rf_pca$table)

# ------------------------------------------------------------------------------
# XGBoost

dtrain <- xgb.DMatrix(data = train_x %>% data.matrix(), label= as.numeric(train_y)-1)
dtrain_up <- xgb.DMatrix(data = train_x_up %>% data.matrix(), label= as.numeric(train_y_up)-1)
dtrain_down <- xgb.DMatrix(data = train_x_down %>% data.matrix(), label= as.numeric(train_y_down)-1)
dtrain_pca <- xgb.DMatrix(data = train_x_pca %>% data.matrix(), label= as.numeric(train_y_pca)-1)

dtest <- xgb.DMatrix(data = test_x %>% data.matrix(), label= as.numeric(test_y)-1)
dtest_pca <- xgb.DMatrix(data = test_x_pca %>% data.matrix(), label= as.numeric(test_y_pca)-1)

xgb_fit_raw = xgboost(data = dtrain, nround = 700, eta=0.05, max_depth=8, objective="binary:logistic")
xgb_fit_up = xgboost(data = dtrain_up, nround = 700, eta=0.05, max_depth=8, objective="binary:logistic")
xgb_fit_down = xgboost(data = dtrain_down, nround = 700, eta=0.05, max_depth=8, objective="binary:logistic")
xgb_fit_pca = xgboost(data = dtrain_pca, nround = 700, eta=0.05, max_depth=8, objective="binary:logistic")

y_predxgb_raw <- predict(xgb_fit_raw, dtest)
y_predxgb_up <- predict(xgb_fit_up, dtest)
y_predxgb_down <- predict(xgb_fit_down, dtest)
y_predxgb_pca <- predict(xgb_fit_pca, dtest_pca)

pred_xgb_raw <- as.numeric(y_predxgb_raw > 0.5)
pred_xgb_up <- as.numeric(y_predxgb_up > 0.5)
pred_xgb_down <- as.numeric(y_predxgb_down > 0.5)
pred_xgb_pca <- as.numeric(y_predxgb_pca > 0.5)

conf_xgb_raw <- confusionMatrix(as.factor(pred_xgb_raw), test_y)
conf_xgb_up <- confusionMatrix(as.factor(pred_xgb_up), test_y)
conf_xgb_down <- confusionMatrix(as.factor(pred_xgb_down), test_y)
conf_xgb_pca <- confusionMatrix(as.factor(pred_xgb_pca), test_y_pca)

confusion_matrix(conf_xgb_raw$table)
confusion_matrix(conf_xgb_up$table)
confusion_matrix(conf_xgb_down$table)
confusion_matrix(conf_xgb_pca$table)

# ------------------------------------------------------------------------------
# DNN
normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

# 정규화
dnn_train_x <- apply(train_x, 2, normalize)
dnn_train_x_up <- apply(train_x_up, 2, normalize)
dnn_train_x_down <- apply(train_x_down, 2, normalize)

dnn_test_x <- apply(test_x, 2, normalize)

# NA <- 0
dnn_train_x[is.na(dnn_train_x)] <- 0
dnn_train_x_up[is.na(dnn_train_x_up)] <- 0
dnn_train_x_down[is.na(dnn_train_x_down)] <- 0

dnn_test_x[is.na(dnn_test_x)] <- 0

dnn_train_y <- to_categorical(train_y)
dnn_train_y_up <- to_categorical(train_y_up)
dnn_train_y_down <- to_categorical(train_y_down)

dnn_test_y <- to_categorical(test_y)
                             
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(44)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2, activation = 'sigmoid')


history <- model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

model %>% fit(
  dnn_train_x, dnn_train_y, 
  epochs = 100, 
  batch_size = 5,
  validation_split = 0.2
)
pred_dnn_raw <- model %>% predict_classes(dnn_test_x, batch_size = 5)
conf_dnn_raw <- confusionMatrix(as.factor(pred_dnn_raw), test_y)

model %>% fit(
  dnn_train_x_up, dnn_train_y_up, 
  epochs = 100, 
  batch_size = 5,
  validation_split = 0.2
)
pred_dnn_up <- model %>% predict_classes(dnn_test_x, batch_size = 5)
conf_dnn_up <- confusionMatrix(as.factor(pred_dnn_up), test_y)

model %>% fit(
  dnn_train_x_down, dnn_train_y_down, 
  epochs = 100, 
  batch_size = 5,
  validation_split = 0.2
)
pred_dnn_down <- model %>% predict_classes(dnn_test_x, batch_size = 5)
conf_dnn_down <- confusionMatrix(as.factor(pred_dnn_down), test_y)

confusion_matrix(conf_dnn_raw$table)
confusion_matrix(conf_dnn_up$table)
confusion_matrix(conf_dnn_down$table)
# ------------------------------------------------------------------------------
# KNN
y_predknn_raw <- knn(train = train_x, test = test_x, cl = train_y, k = 1)
y_predknn_up <- knn(train = train_x_up, test = test_x, cl = train_y_up, k = 1)
y_predknn_down <- knn(train = train_x_down, test = test_x, cl = train_y_down, k = 1)

conf_knn_raw <- confusionMatrix(y_predknn_raw, test_y)
conf_knn_up <- confusionMatrix(y_predknn_up, test_y)
conf_knn_down <- confusionMatrix(y_predknn_down, test_y)

confusion_matrix(conf_knn_raw$table)
confusion_matrix(conf_knn_up$table)
confusion_matrix(conf_knn_down$table)
# ------------------------------------------------------------------------------
# OC-SVM
# 처음부터 끝까지 같은 값일 경우 scale 불가하여 제외
ibp_feature_svm <- ibp_feature_df %>% select(-imvalue_fn10)
data_TRUE = subset(ibp_feature_svm, event==0)
data_FALSE = subset(ibp_feature_svm, event==1)

inTrain = createDataPartition(1:nrow(data_TRUE), p=0.8, list=FALSE)

train_svm = data_TRUE[inTrain,]
train_x_svm = train_svm %>% select(-event)
train_y_svm = train_svm$event

test_TRUE = data_TRUE[-inTrain, ]
test_svm = rbind(test_TRUE, data_FALSE)
test_x_svm = test_svm %>% select(-event)
test_y_svm = test_svm$event

svm.model = svm(train_x_svm, y=NULL,
                type='one-classification',
                nu=0.10,
                scale = TRUE,
                kernel = "radial")

y_predsvm_raw <- predict(svm.model, test_x_svm)
y_predsvm_raw <- ifelse(y_predsvm_raw == "TRUE", 0, 1)
conf_svm_raw <- confusionMatrix(as.factor(y_predsvm_raw), test_y_svm)

confusion_matrix(conf_svm_raw$table)

# ------------------------------------------------------------------------------
# 10-fold 검증
cv_model = xgb.cv(data = data.matrix(train_x), label = as.numeric(train_y)-1, num_class = levels(train_y) %>% length,
                  nfold = 10, nrounds = 700, early_stopping_rounds = 150, objective="binary:logistic", verbose = F, prediction = T)
pred_df = cv_model$pred %>% as.data.frame %>%
  mutate(pred = levels(train_y)[max.col(.)] %>% as.factor, actual = train_y)
pred_df %>% select(pred, actual) %>% table

conf_xgbcv = confusionMatrix(pred_df$pred, pred_df$actual)
confusion_matrix()
