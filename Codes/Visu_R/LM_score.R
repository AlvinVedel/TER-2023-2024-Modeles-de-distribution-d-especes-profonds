
library(fpp3)

data <- read.csv(file="data/matrice_voisins.txt")

str(data)
data$voisins <- as.factor(data$voisins)

#Evolution du score en fonction du
ggplot(data, aes(x = proba, y = score, color = as.factor(voisins))) +
  geom_point() + geom_line()+
  labs(x = "Proba", y = "Score", color = "Voisins") +
  scale_color_discrete(name = "Voisins")

#Score moyen en  fonction de proba 

resultats_agreges <- aggregate(data$score, list(voisins = data$voisins), mean)
ggplot(resultats_agreges, aes(x = voisins, y = x)) +
  geom_point()+geom_line(group=1)+
  labs(x = "Voisins", y = "Moyenne du Score")

#Score moyen en  fonction de proba 
plot(aggregate(data$score, list(proba = data$proba), mean))
lines(aggregate(data$score, list(proba = data$proba), mean))

#Modèle
mod <- lm(score~voisins,data = data)
summary(mod)
