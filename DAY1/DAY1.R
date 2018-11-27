library(caret)
library(e1071)
dataset<-iris
head(dataset)
dataset$Species

t1<-train(Species~.,data=dataset,method="knn")
predict(t1,dataset[1,(1:4)])

#---------------------------------------------------------------------------------------

v<-createDataPartition(dataset$Species,p=0.8,list=FALSE)
length(v)
trainData<-dataset[v,]
testData<-dataset[-v,]
dim(trainData)
dim(testData)

t2<-train(Species~.,data=trainData,method="knn")
t2
pred<-predict(t2,testData)

confusionMatrix(testData[,5],pred)

#----------------------------------------------------------------------------------------
library(caret)
library(e1071)
e<-read.table(file.choose())
e
head(e)
t3<-train(V9~.,data=e[,-1],method="knn")
predict(t3,e[1,(2:8)])

#---------------------------------------------------------------------------------------

library(caret)
library(e1071)
g<-read.table(file.choose())
g
head(g)
t3<-train(V6~.,data=g[,-1],method="knn")
predict(t3,g[1,(1:8)])

