install.packages("tidyr", repos = "https://cloud.r-project.org")
install.packages("dplyr", repos = "https://cloud.r-project.org")
library("tidyr")
library("dplyr")
orig_data <- read.csv("filler_wideform_data.csv")
dim(orig_data)
fp_data <- orig_data %>% pivot_longer(cols=15:91,names_to="subject",values_to="fp")#0 idx?YES
head(fp_data)
fp_data[15:91]<-NULL#remove wideform columns
dim(fp_data)
#fp_data[15:18]<-NULL#remove the later columns
ro_data <- orig_data %>% pivot_longer(cols=92:168,names_to="subject",values_to="ro")#0 idx?YES
head(ro_data)
ro_data[15:91]<-NULL#remove wideform columns

#ro_data[15:18]<-NULL#remove the later columns
ro_data$subject<-gsub("ro","fp",ro_data$subject)
head(fp_data$subject)
head(ro_data$subject)


merged_data <- right_join(fp_data,ro_data)
dim(merged_data)
write.csv(merged_data,"filler_longform_data.csv")
