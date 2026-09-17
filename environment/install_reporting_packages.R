packages<-c("dplyr","tidyr","ggplot2","patchwork","scales","knitr","rmarkdown")
missing<-packages[!vapply(packages,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) install.packages(missing,repos="https://cloud.r-project.org")
message("Core reporting packages available. magick is optional for original-image galleries.")
