#Autoencoder2

library(keras)
otutable <- read.csv("C:/Users/dell/Downloads/otutable-sub2k/abundance-Soil-non-saline.tsv", sep = '\t', header = TRUE, row.names = "seqID")
x_train <- otutable
x_train <- as.matrix(x_train)
#x_train <- t(x_train)

input_size = ncol(x_train)

enc_input = layer_input(shape = input_size)
enc_output = enc_input %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 3)

encoder = keras_model(enc_input, enc_output)
summary(encoder)

dec_input = layer_input(shape = 3)
dec_output = dec_input %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = input_size, activation = "sigmoid")

decoder = keras_model(dec_input, dec_output)
summary(decoder)

aen_input = layer_input(shape = input_size)
aen_output = aen_input %>% 
  encoder() %>% 
  decoder()

aen = keras_model(aen_input, aen_output)
summary(aen)

aen %>% compile(optimizer="rmsprop", loss="mean_squared_error")

aen %>% fit(x_train,x_train, epochs=200, batch_size=128) #More epochs to run, loss is still decreasing

encoded = encoder %>% predict(x_train)

#--------------------------------PLOT 3D---------------------------------------
install.packages("scatterplot3d")
library(scatterplot3d)

s3d <-scatterplot3d(encoded[,1],encoded[,2],encoded[,3], pch=16, highlight.3d=TRUE,
                    type="h", main="5 layers new")

#PRINT WITH COLOR
DF <- data.frame(x = encoded[,1], y = encoded[,2], z = encoded[,3], group = seq.int(nrow(encoded)))
s3d <- with(DF, scatterplot3d(x, y, z, color = as.numeric(group), pch = 19, main = "5 layers colored"))


#legend(s3d$xyz.convert(0.5, 0.7, 0.5), pch = 19, yjust=0,
#       legend = levels(DF$group), col = seq_along(levels(DF$group)))



