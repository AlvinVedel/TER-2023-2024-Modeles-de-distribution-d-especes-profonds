
#Données géographique

setwd("C:/Users/mbrei/Desktop/MIASHS/TER/R/data")
abio <- read.csv2(file="enviroTab_pa_train.csv")

library(sf)
place_sf <- st_as_sf(abio, coords = c("lon", "lat"), crs = 4326)
place_sf

library(maptiles)
osm <- get_tiles(x = place_sf, zoom = 7)
plot_tiles(osm)
plot(st_geometry(place_sf), pch = 4, cex = 2, col = "red", add = TRUE)







                                                    #SDM

library(data.table)
library(raster)
library(randomForest)
library(lattice)
library(RColorBrewer)
library(PresenceAbsence)

#Appel de la donnée
setwd("C:/Users/mbrei/Desktop/MIASHS/TER/R/data")
abio <- read.csv2(file="enviroTab_pa_train.csv")

#Modif
summary(abio)
str(abio)
abio$bio3 <- as.numeric(abio$bio3)

#Data importante 
abio_cols <- c('patchID','bio1','bio2', 'bio3', 'bio4', 'bio5')
abio_df <- data.frame(abio)[abio_cols]
summary(abio_df)






#Création de raster : EXEMPLE
library(raster)

# R de GIT :
avi_dat <- read.csv(file = "data/Data_SwissBreedingBirds.csv")
avi_cols <- c('Turdus_torquatus', 'bio_5', 'bio_2', 'bio_14', 'std', 'rad', 'blockCV_tile')
avi_df <- data.frame(avi_dat)[avi_cols]
summary(avi_df)

bio_curr <- getData('worldclim', var='bio', res=0.5, lon=5.5, lat=45.5, path='data')[[c(2,5,14)]]
bio_fut <- getData('CMIP5', var='bio', res=0.5, lon=5.5, lat=45.5, rcp=45, model='NO', year=50, path='data', download=T)[[c(2,5,14)]]

# A spatial mask of Switzerland in Swiss coordinates
bg <- raster('/vsicurl/https://damariszurell.github.io/SDM-Intro/CH_mask.tif')

# The spatial extent of Switzerland in Lon/Lat coordinates is roughly:
ch_ext <- c(5, 11, 45, 48)

# Crop the climate layers to the extent of Switzerland
bio_curr <- crop(bio_curr, ch_ext)

# Re-project to Swiss coordinate system and clip to Swiss political bounday
bio_curr <- projectRaster(bio_curr, bg)
bio_curr <- resample(bio_curr, bg)
bio_curr <- mask(bio_curr, bg)
names(bio_curr) <- c('bio_2', 'bio_5', 'bio_14')

# For storage reasons the temperature values in worldclim are multiplied by 10. For easier interpretability, we change it back to °C.
bio_curr[[1]] <- bio_curr[[1]]/10
bio_curr[[2]] <- bio_curr[[2]]/10

# Repeat above steps for future climate layers
bio_fut <- crop(bio_fut, ch_ext)
bio_fut <- projectRaster(bio_fut, bg)
bio_fut <- resample(bio_fut, bg)
bio_fut <- mask(bio_fut, bg)
names(bio_fut) <- c('bio_2', 'bio_5', 'bio_14')
bio_fut[[1]] <- bio_fut[[1]]/10
bio_fut[[2]] <- bio_fut[[2]]/10











