#Autoencoder2

library(keras)
otutable <- read.csv("C:/Users/dell/Downloads/otutable-sub2k/abundance-Soil-non-saline.tsv", sep = '\t', header = TRUE, row.names = "seqID")
x_train <- otutable
x_train <- as.matrix(x_train)

input_size = ncol(x_train)

enc_input = layer_input(shape = input_size)
enc_output = enc_input %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 2)

encoder = keras_model(enc_input, enc_output)
summary(encoder)

dec_input = layer_input(shape = 2)
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

aen %>% fit(x_train,x_train, epochs=50, batch_size=128) 

encoded = encoder %>% predict(x_train)

#PRINT WITH COLOR
DF <- data.frame(x = encoded[,1], y = encoded[,2], group = seq.int(nrow(encoded)))
#s3d <- with(DF, scatterplot3d(x, y, z, color = as.numeric(group), pch = 19, main = "5 layers colored"))
library(ggplot2)
ggplot(DF, aes(x,y), color = as.numeric(group)) + geom_point(color = as.numeric(DF[,3])) + ggtitle("5 layers 2D")



