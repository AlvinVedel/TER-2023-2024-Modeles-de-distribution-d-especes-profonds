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
table$`Id de l'espèce`

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


#On s'intéresse aux coordoonées, patchID et lat/long
str(data)
patch <- as.factor(data$patchID)

class(patch)
length(unique(patch))

data_patch <- data.frame(table(patch))

colnames(data_patch) <- c("PatchID", "Fréquence")

# Triez le dataframe par le nombre d'apparitions, de la plus vue à la moins vue
data_patch <- data_patch[order(-data_patch$Fréquence), ]

table(data$dayOfYear,data$patchID) # NULL

#DatofYear
data.frame(table(data$dayOfYear))


hist(data$dayOfYear)

data_year <- data.frame(DayofYear=data$dayOfYear,Year=data$year)

# Créez un graphique à barres groupées avec ggplot2
ggplot(data_year, aes(x = DayofYear, fill=factor(Year))) +
  geom_histogram(alpha=0.5,position="stack",colour="black") + 
  labs(title="Nombre d'observations en fonction du jour de l'année",x = "Jour de l'année", y = "Efectif", fill="Année")+
  scale_fill_manual(values=c("#000000","#000060","#004EFF","#00BCFF","#C0C0C0"))+
  geom_vline(xintercept = c(80, 170, 266, 353), color = "red") +
  annotate("text", x = c(120,200,300,20), y = 8000, 
           label = c("Printemps", "Été", "Automne","Hiver"), color = "red")

                      
