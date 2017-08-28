  result32batch<-as.data.frame(resultbatch300)


Accuracy300 <- result32batch[grep("Accuracy",result32batch[,1]), ]

Precision300<- result32batch[grep("precision:",result32batch[,1]), ]

auc_score300<- result32batch[grep("roc_auc_score",result32batch[,1]), ]

Recall300 <- result32batch[grep("Recall:",result32batch[,1]), ]

mean(Precision[,2])
mean(Precision200[,2])  
mean(Precision300[,2])

mean(Recall[,2])
mean(Recall200[,2])  
mean(Recall300[,2])  
  
plot(Precision[,2],type='p', xlab="N-th model run",ylab="Precision",main="Precisions of the model after 100 runs")
points(Precision200[,2],col="red")  
points(Precision300[,2],col="blue")  
abline(h=0.6765 ,col="green")
legend(75,0.77,c("Model-Batch-size:32","Model-Batch-Size :200","Model-Batch-size :300","Precision of the Original model"),cex=0.6,pch=15,col=c("black","red","blue","green"))
boxplot(Precision[,2], xlab="N-th model run",ylab="Precision",main="Precisions of the model after 100 runs")  
  
plot(Recall[,2],type='p', xlab="N-th model run",ylab="Recall",main="Recall of the model after 100 runs")
points(Recall200[,2],col="red")  
points(Recall300[,2],col="blue")  
abline(h=0.7264 ,col="green")
  legend(75,0.64,c("Model-Batch-size:32","Model-Batch-Size :200","Model-Batch-size :300","Precision of the Original model"),cex=0.6,pch=15,col=c("black","red","blue","green"))  
  
# smooth curves
  
x<-1:100
  y1 <- Precision[,2]
  lo <- loess(y1~x)
  plot(x,y1)
  lines(predict(lo), col='red', lwd=2)
  y2 <- Precision200[,2]
  lo <- loess(y2~x)
  plot(x,y2)
  lines(predict(lo), col='blue', lwd=2)
  y3 <- Precision300[,2]
  lo <- loess(y3~x)
  plot(x,y3)
  lines(predict(lo), col='red', lwd=2)
  
  
result200batch<-as.data.frame(resultbatch200)


Accuracy <- result32batch[grep("Accuracy",result32batch[,1]), ]

Precision<- result32batch[grep("precision:",result32batch[,1]), ]

auc_score<- result32batch[grep("roc_auc_score",result32batch[,1]), ]

Recall <- result32batch[grep("Recall:",result32batch[,1]), ]



accuracy<-{}
for (i in 1:400){

  #accuracy<-result32batch[1,1]
  accuracy1<-result32batch[i+3*(i-1),2]
  
  accuracy<-append(accuracy,accuracy)
}