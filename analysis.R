install.packages("lme4", repos = "https://cloud.r-project.org")
library(lme4)
filler_data = read.csv("filler_longform_data.csv")
str(filler_data)
filler_data$subject <- as.factor(filler_data$subject)
#TODO: At some point you should train test split
filler.model_surp = lmer(data=filler_data,
                           fp ~ sum_surprisal+spillover_1+spillover_2+word_pos+lg_freq*len+(1 | subject),
control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
summary(filler.model_surp)

filler.model_nosurp = lmer(data=filler_data,
                         fp ~ word_pos+lg_freq*len+
                           (1 | subject),
control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
summary(filler.model_nosurp)

target_data = read.csv("target_interest_items.csv")
target_data$subject <- as.factor(target_data$subject)

levels(target_data$subject) == levels(filler_data$subject)


#summary(target_data[target_data$ambiguity=="True",]$sum_surprisal)
#summary(target_data[target_data$ambiguity=="False",]$sum_surprisal)

surp_ambig_pred <- predict(filler.model_surp,newdata=target_data[target_data$ambiguity=="True",],allow.new.levels=TRUE)
nosurp_ambig_pred <- predict(filler.model_nosurp,newdata=target_data[target_data$ambiguity=="True",],allow.new.levels=TRUE)
surp_unambig_pred <- predict(filler.model_surp,newdata=target_data[target_data$ambiguity=="False",],allow.new.levels=TRUE)
nosurp_unambig_pred <- predict(filler.model_nosurp,newdata=target_data[target_data$ambiguity=="False",],allow.new.levels=TRUE)
print("AMBIGUOUS, SURP MODEL:")
summary(surp_ambig_pred)
print("AMBIGUOUS, NO_SURP MODEL:")
summary(nosurp_ambig_pred)
print("UNAMBIGUOUS, SURP MODEL:")
summary(surp_unambig_pred)
print("UNAMBIGUOUS, NOSURP MODEL:")
summary(nosurp_unambig_pred)
