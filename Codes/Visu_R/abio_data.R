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
describe(abio_data)

#CHangement de type
abio_data$bio3 <- as.numeric(abio_data$bio3)

#ACP
library(FactoMineR)

acp_data <- data.frame(abio_data[1],abio_data[,10:28])

resultat=PCA(data.frame(abio_data[1],abio_data[,10:28]), graph=F)
resultat
summary(resultat)
abio_data$year[is.na(abio_data$year)]

plot.PCA(resultat, choix="ind", label="all", habillage="none")

#Analyse
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
ggplot(abio_data, aes(x=bio1,group=year))+geom_bar()



#Doublons patchID
patchfreq <- data.frame(table(abio_data$patchID))
patchfreq[919,]

patchdoublon <- patchfreq$Var[which(patchfreq$Freq!=1)]

dataclean <- subset(abio_data, !(patchID %in% patchdoublon))

acp <- PCA(acp_data, quali.sup=1,graph=T)
plot(acp, label='none')


#TEST AFD
source(file = "C:/Users/mbrei/Desktop/MIASHS/S1/AnalyseMultiple/AFD.R")
X <- subset(acp_data,select=-1)
y <-acp_data[,1]
AFD(X,y, "FR")




