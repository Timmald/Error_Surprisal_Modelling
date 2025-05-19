install.packages("tidyr", repos = "https://cloud.r-project.org")
install.packages("dplyr", repos = "https://cloud.r-project.org")
library("tidyr")
library("dplyr")
orig_data <- read.csv("filler_means.csv")
fp_data <- orig_data %>% pivot_longer(cols=17:93,names_to="subject",values_to="fp")#0 idx?
fp_data[17:93]<-NULL
fp_data[18:21]<-NULL
ro_data <- orig_data %>% pivot_longer(cols=94:170,names_to="subject",values_to="ro")#0 idx?
ro_data[17:93]<-NULL
ro_data[18:21]<-NULL
ro_data$subject<-gsub("ro","fp",ro_data$subject)
head(fp_data$subject)
head(ro_data$subject)


merged_data <- right_join(fp_data,ro_data)
write.csv(merged_data,"filler.pivot.csv")
