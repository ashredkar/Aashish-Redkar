## Final Project on HR

library(dplyr)          # used for glimpse and mutate
library(car)            # used for vif calculation
library(tree)           # Dtree & RF
library(ISLR)           # Dtree & RF
library(randomForest)   # Dtree & RF
library(xgboost)        # used for XGBoost
library(ggplot2)        # used for Plots - Visualisation
library(pROC)		        # used for roc auc
library(tidyr)	      	# view the graph showing f1, precision, etc
library(gbm)            # used for GBM

setwd("C:/Users/milind/Documents/Aashish Redkar/Eduvancer/R/Projects/4 Human Resources")

#  Read Data and combine files of train & test
hr_train=read.csv("hr_train.csv")
hr_test=read.csv("hr_test.csv")
hr_train$source="train"
hr_test$source="test"
hr_test$left="NA"
hr_all=rbind(hr_train,hr_test)



glimpse(hr_all)
# converted to Numerical

hr_all=hr_all%>%
  mutate(left=ifelse(left=="NA",NA,as.numeric(left)))

glimpse(hr_all)

table(hr_all$sales)

hr_all=hr_all%>%
  mutate(sales_hr=as.numeric(sales=="hr"),
         sales_ac=as.numeric(sales=="accounting"),
         sales_rand=as.numeric(sales=="RandD"),
         sales_mkt=as.numeric(sales=="marketing"),
         sales_mng=as.numeric(sales=="product_mng"),
         sales_it=as.numeric(sales=="IT"),
         sales_support=as.numeric(sales=="support"),
         sales_tech=as.numeric(sales=="technical"),
         sales_sales=as.numeric(sales=="sales"))%>%
  select(-sales)


table(hr_all$salary)

hr_all=hr_all%>%
  mutate(low=as.numeric(salary=="low"),
         medium=as.numeric(salary=="medium"))%>%
  select(-salary)

glimpse(hr_all)

unlist(lapply(hr_all,function(x) sum(is.na(x))))

hr2_train = hr_all %>% 
  filter(source=="train") %>% 
  select(-source)
hr2_test = hr_all %>% 
  filter(source =="test") %>% 
  select(-source,-left)

set.seed(9)
s=sample(1:nrow(hr2_train),0.7*nrow(hr2_train))
hrm_train=hr2_train[s,]
hrm_test=hr2_train[-s,]


gbm.fit=gbm(left~.,
            data=hrm_train,
            distribution = "gaussian",
            n.trees = 100,interaction.depth = 3)

summary(gbm.fit)
View(hrm_train)
train.predicted=predict.gbm(gbm.fit,newdata=hrm_train,n.trees=100)

train.predicted=predict.gbm(gbm.fit,newdata=hrm_train,n.trees=100)
(train.predicted-hrm_train$left)**2 %>% mean() %>% sqrt()

test.predicted=predict.gbm(gbm.fit,newdata=hrm_test,n.trees=100)

test.predicted=predict.gbm(gbm.fit,newdata=hrm_test,n.trees=100)
(test.predicted-hrm_test$left)**2 %>% mean() %>% sqrt()


hrm_train$gbm_predscore=predict(gbm.fit,newdata=hrm_train,type="response")
View(hrm_train[,c("left","gbm_predscore")])

cutoff=0.3

pred_gbm=as.numeric(hrm_train$gbm_predscore>cutoff)
temp_gbm=hrm_train[,c("left","gbm_predscore")]

View(cbind.data.frame(temp_gbm,pred_gbm))

temp1_gbm=(cbind.data.frame(temp_gbm,pred_gbm))
table(temp1_gbm$left,temp1_gbm$pred_gbm)

TP1=sum(pred_gbm==1 & hrm_train$left==1)
FP1=sum(pred_gbm==1 & hrm_train$left==0)
FN1=sum(pred_gbm==0 & hrm_train$left==1)
TN1=sum(pred_gbm==0 & hrm_train$left==0)

P1=TP1+FN1
N1=TN1+FP1

total1=P1+N1

cutoff_data1=data.frame(cutoff=0,TP1=0,FP1=0,FN1=0,TN1=0)
cutoffs1=round(seq(0,1,length=100),3)

for (cutoff in cutoffs1){
  pred_gbm=as.numeric(hrm_train$gbm_predscore>cutoff)
  TP1=sum(pred_gbm==1 & hrm_train$left==1)
  FP1=sum(pred_gbm==1 & hrm_train$left==0)
  FN1=sum(pred_gbm==0 & hrm_train$left==1)
  TN1=sum(pred_gbm==0 & hrm_train$left==0)
  cutoff_data1=rbind(cutoff_data1,c(cutoff,TP1,FP1,FN1,TN1))
}

cutoff_data1=cutoff_data1[-1,]
View(cutoff_data1)
cutoff_data1=cutoff_data1 %>%
  mutate(Sn=TP1/P1, 
         Sp=TN1/N1,
         Precision=TP1/(TP1+FP1),
         F1=(2*Precision*Sn)/(Precision+Sn),
         dist=sqrt((1-Sn)**2+(1-Sp)**2),
         P=FN1+TP1,N1=TN1+FP1) %>%
  mutate(KS=abs((TP1/P1)-(FP1/N1))) %>%
  mutate(Accuracy=(TP1+TN1)/(P1+N1)) %>%
  mutate(Lift=(TP1/P1)/((TP1+FP1)/(P1+N1))) %>%
  mutate(M=(8*FN1+2*FP1)/(P1+N1))

View(cutoff_data1)

cutoff_viz1=cutoff_data1 %>%
  select(cutoff,Sn,Sp,dist,KS,Accuracy,Lift,M) %>%
  gather(Criterion,Value,Sn:M) 

ggplot(filter(cutoff_viz1,Criterion!="Lift"),aes(x=cutoff,y=Value,color=Criterion))+
  geom_line()

KS_cutoff1=cutoff_data1$cutoff[which.max(cutoff_data1$KS)][1]
KS_cutoff1

hrm_test$predscore=predict(gbm.fit,newdata = hrm_test,type = "response")

table(hrm_test$left,as.numeric(hrm_test$predscore>KS_cutoff1))

# Filter records where predicted is 1
filtered1 = hrm_test[hrm_test$pred_test==1,]


roccurve=roc(hrm_train$left~hrm_train$gbm_predscore)
plot(roccurve)
auc(roccurve)

roc_data=cutoff_data1 %>% 
  select(cutoff,Sn,Sp) %>% 
  mutate(TPR=Sn,FPR=1-Sp) %>% 
  select(cutoff,TPR,FPR)

ggplot(roc_data,aes(x=FPR,y=TPR))+geom_line()+ggtitle("My ROC Curve")




hr2_test$predscore=predict(gbm.fit,newdata = hr2_test,type = "response")
table(hr2_test$left,as.numeric(hr2_test$predscore>KS_cutoff1))
ans=data.frame(hr2_test,hr2_test$predscore>KS_cutoff1)
View(ans)

ans=ans%>%
  mutate(prediction=ifelse(hr2_test.predscore...KS_cutoff1=="TRUE",1,0))

write.csv(ans,file="hr_project_Aashish_Redkar.csv")         
