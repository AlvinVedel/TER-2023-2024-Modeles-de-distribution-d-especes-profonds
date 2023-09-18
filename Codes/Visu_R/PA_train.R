                # PA Train 
#Librairie
library(fpp3)


#Appel de la donnée
setwd("C:/Users/mbrei/Desktop/MIASHS/TER/R/data")

data <- read.csv2(file="Presences_Absences_train.csv")

#Première visualition du df
head(data)
summary(data)
str(data)

#Visualisation de la fréquence des espèces
hist(data$speciesId,freq=T,breaks=1000)

table <- data.frame(sort(table(data$speciesId),decreasing=T)[1:100])
colnames(table) <- c(var1="Id de l'espèce",Freq="Nombre d'observations") 

#Tri, espèce la plus présente 
species1 <- data[which(data$speciesId==4284),]

species1$date <- as.Date(data[which(data$speciesId==4284),]$date, format = "%d/%m/%Y")

ggplot(species1, aes(x = date)) +
  geom_histogram(binwidth = 50, fill = "white", color = "black") +
  labs(title = "Evolution de nomvbre d'observations en fonction de l'année de l'espèce 4284",
       x = "Date",
       y = "Nombre d'observations")



# Données cartographique
library(ggmap)
dev.off()

coord_sp_1 <- data.frame(x=species1$lat,y=species1$lon)  #coord de l'espèce
coord_sp_1 <- na.omit(coord_sp_1)


ggplot(coord_sp_1, aes(x = x, y = y)) +
  geom_point() +
  labs(title = "Répartition de l'espèce", x = "Longitude", y = "Latitude")+
  scale_x_discrete(max(coord_sp_1$x))+
  scale_y_discrete(max(coord_sp_1$y))

help(scale_x_discrete)
#carte de la france








