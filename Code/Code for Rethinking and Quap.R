# in order to insall all package, install rtools from
# https://cran.r-project.org/bin/windows/Rtools/
# install Rtools35.exe

install.packages("usethis")
library(usethis)
install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
library(rstan)
install.packages(c("coda","mvtnorm","devtools"))
library(devtools)
devtools::install_github("rmcelreath/rethinking",ref="Experimental")
library(rethinking)

?quap


#lets try again
