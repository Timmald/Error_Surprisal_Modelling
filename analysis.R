install.packages("lme4", repos = "https://cloud.r-project.org")
library(lme4)
library(ggplot2)
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

print("AMBIGUOUS SENTENCES: WOI SURPRISAL")
summary(target_data[target_data$ambiguity=="True",]$sum_surprisal)
print("UNAMBIGUOUS SENTENCES: WOI SURPRISAL")
summary(target_data[target_data$ambiguity=="False",]$sum_surprisal)


print("------------------------")

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


results = data.frame(ambiguity = rep("Ungrammatical",length(surp_ambig_pred)),pred = surp_ambig_pred)
results2 = data.frame(ambiguity = rep("Grammatical",length(surp_unambig_pred)),pred = surp_unambig_pred)
results<-rbind(results,results2)


target_data$item_diffs <- target_data[target_data$ambiguity=="True",]$sum_surprisal-target_data[target_data$ambiguity=="False",]$sum_surprisal
pretty_for_plotting = target_data$ambiguity <- factor(target_data$ambiguity, levels = c("True", "False"), labels = c("Ungrammatical", "Grammatical"))
ggplot(data=target_data, aes(x=item_diffs)) +
    geom_density()+
    geom_vline(data=target_data,aes(xintercept = mean(item_diffs)))+
    labs(title="Difference in WOI Surprisal over Grammaticality",x="Difference (bits)",y="Density")

ggplot(data=results, aes(x=pred,group=ambiguity,fill=ambiguity)) +
    geom_density(alpha=0.4)+
    labs(title="Distributions of Predicted FP time",x="Predicted FP (ms)",y="Density",fill="Grammaticality")
