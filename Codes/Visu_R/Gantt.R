
# GANTT
library(vistime)
dat <- data.frame(Names=c("Alvin", "Flo", "Matis","Alvin","Alvin"),
                  Tasks = c("Gestion de Projet", "Cartographie","Exploration de données","Modèle constant","Matrice co-occurence","Modeles PO","Gestion de projet"),
                  start = c("2023-09-18 16:00:00", "2023-09-18 16:00:00","2023-09-18 16:00:00","2023-09-18 16:00:00"),
                  end = c("2023-09-18 19:00:00","2023-09-18 19:00:00","2023-09-18 19:00:00"),
                  color = c("#3e6b80", "#af7a6d", "#84a59d"),
                  fontcolor = rep("white", 3))
vistime(dat, events="Names", groups="Tasks", title="Gantt chart")


Flo : carto ; Knn ; Knn et probas
Alvin : Gestion de projet ; Modeles constants ; Matrice co-occurence ; Modeles PO
