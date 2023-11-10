
#Données
setwd("C:/Users/mbrei/Desktop/MIASHS/TER/R")
abio <- read.csv2(file="data/enviroTab_pa_train.csv",sep=";",dec=".")
patrain <- read.csv2(file="data/Presences_Absences_train.csv")

#Occurence du pa train
patrain$patchID <- as.factor(patrain$patchID)
patrain$speciesId <- as.factor(patrain$speciesId)
str(patrain)

#occurence de abio
abio$patchID <- as.factor(abio$patchID)
str(abio)

#DOublons abio
abio$patchID[which(duplicated(abio$patchID)==TRUE)]  #800 doublons
abio[which(abio$patchID==3000219),]  #Même données abio

library(tidyverse)
abio <- distinct(abio, patchID, .keep_all=TRUE)

#Tri des tableaux
pa2 <- data.frame(patchID=patrain$patchID,speciesId=patrain$speciesId)
abioclean <- cbind(patchID=abio[,1],abio[,10:28])

pa2$patchID <- as.factor(pa2$patchID)
pa2$speciesId <- as.factor(pa2$speciesId)
str(pa2)

#Merge
abiomergetous <- merge(abio, patrain, by = "patchID")

abiomerge <- merge(abioclean, pa2, by = "patchID")
abiomerge <- abiomerge[,-1]
str(abiomerge)

#Tbleau sortir
library(openxlsx)
write.xlsx(abiomergetous,"C:/Users/mbrei/Desktop/MIASHS/TER/R/data")

#Random Forest
library(randomForest)
model <- randomForest(speciesId ~., data = abiomerge, ntree = 10, na.action = na.omit)
summary(model)

# individu out of the bag, taux faible modèle juste 
model$oob.times
hist(model$oob.times)

#predict 
patest <- read.csv2(file = "data/enviroTab_pa_test.csv")

patest$predicted <- predict(model, patest[,11:29])

library(caret)
conf <- confusionMatrix(data = patest$predicted)

conf$byClass

