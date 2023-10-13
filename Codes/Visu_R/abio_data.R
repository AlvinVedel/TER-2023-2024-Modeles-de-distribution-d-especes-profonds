#Librairie
library(fpp3)


#Appel de la donnée
setwd("C:/Users/mbrei/Desktop/MIASHS/TER/R/data")

abio_data <- read.csv2(file="enviroTab_pa_train.csv")
pa_data <- read.csv2(file="Presences_Absences_train.csv")


#Première visualition du df
head(abio_data)
str(abio_data)
abio_data$patchID <- as.factor(abio_data$patchID)

library(psych)
describe(data$patchID)

#CHangement de type
abio_data$bio3 <- as.numeric(abio_data$bio3)

#ACP
library(FactoMineR)
resultat=PCA(abio_data[,10:28], graph=F)
resultat

abio_data$year[is.na(abio_data$year)]

plot.PCA(resultat, choix="ind", label="none", habillage="none")

#Analyse
cor(resultat)

resultat$eig[2,3]

resultat$var$cor

#Explor
library(explor)
resultat2=PCA(abio_data[,10:28], graph=FALSE)
explor::explor(resultat2)

#Espèces associées au patchID
pa_data$speciesId[pa_data$patchID==55880]




#Analyse de bio
date <- as.Date(paste(abio_data$year, "/01/01", sep = ""), format = "%Y/%m/%d") + (abio_data$dayOfYear - 1)











