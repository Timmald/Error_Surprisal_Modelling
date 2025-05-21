install.packages("ggplot2", repos = "https://cloud.r-project.org")
install.packages("lme4", repos = "https://cloud.r-project.org")
library(ggplot2)
library(lme4)

dat = read.csv("filler_spillover.pivot.csv",header=TRUE)
hist(dat$fp)
noReg = dat[dat$ro==0 & dat$fp<=1200, ]
reg = dat[dat$ro==1 & dat$fp<=1200, ]
regnoNA = na.omit(reg)
print(nrow(regnoNA))
noRegmod <- lmer(log(fp) ~ len*lg_freq+ sum_surprisal*ro+spillover_1+spillover_2+(1 | subject),data = noReg,REML=FALSE)
regmod <- lmer(log(fp) ~ len*lg_freq+ sum_surprisal*ro+spillover_1+spillover_2+(1 | subject),data = reg,REML=FALSE)

summary(noRegmod)
summary(regmod)

resnoreg <- residuals(noRegmod, "pearson", scaled=TRUE)
resreg <- residuals(regmod, "pearson", scaled=TRUE)
prednoreg <- predict(noRegmod,newdata=na.omit(noReg))
predreg <- predict(regmod,newdata=na.omit(reg))
pltDFreg <- data.frame(res=resreg,fitted=predreg)
pltDFnoreg <- data.frame(res=resnoreg,fitted=prednoreg)



qqnorm(residuals(noRegmod, "pearson", scaled=TRUE))
hist(residuals(noRegmod, "pearson", scaled=TRUE))
qqnorm(residuals(regmod, "pearson", scaled=TRUE))
hist(residuals(regmod, "pearson", scaled=TRUE))

summary(pltDFreg[pltDFreg$fitted >= 5.25,])
summary(pltDFnoreg[pltDFnoreg$fitted >= 5.25,])
ggplot(data = pltDFnoreg, aes(x = fitted, y = res)) +
geom_point(color = "blue", size = .25, alpha = 0.7)

ggplot(data = pltDFreg, aes(x = fitted, y = res)) +
geom_point(color = "blue", size = .25, alpha = 0.7)

#logistic.model <- glmer(ro ~ len*lg_freq+ spillover_1+spillover_2+sum_surprisal+(1 | subject),family = 'binomial', data = dat)
#(logistic.model)

#model <- lm(dat$fp_mean ~ exp(dat$len)+dat$lg_freq+dat$mean_surprisal+dat$len*dat$lg_freq+dat$ro_mean)
#nolen <- lm(dat$fp_1 ~ dat$lg_freq+dat$mean_surprisal+dat$ro_1)

#predictions <- predict(model,dat)

#newDat <- dat

#newDat$pred = predictions

#summary(model)

#boxplot(dat$fp_2-predictions~dat$len)
#
#ggplot(data = newDat, aes(x = mean_surprisal, y = dat$fp_2-pred)) +
#geom_point(color = "blue", size = .25, alpha = 0.7)
#geom_smooth(method = "lm")