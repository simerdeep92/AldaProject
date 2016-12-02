#install.packages("rgl")
#http://stackoverflow.com/a/22213565/2433023i

#Reference: http://stackoverflow.com/a/8013351/2433023
library(rgl)
x = rnorm(100)
y = rnorm(100)
z = rnorm(100)
r = z
r = cut(r, breaks=64)
cols = rainbow(64)[as.numeric(r)]
#plot3d(pr$rotation[,1:3], col = cols)
m <- seq(-5, c, by=0.5)
n <- seq(-5, g, by=0.5)
plot3d(m, n, accuracy, col = cols)

scaling <- plot3d(m, n, accuracy)
scaling
# In addition you may add a line to the plot with these two lines;
a <- scaling(c(-5,5), c(0,0), c(0,0))
lines3d(scaling$m, scaling$n, scaling$accuracy, col = "blue", size = 2) 
