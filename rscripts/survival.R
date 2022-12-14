library(glue)
library(survival)

args = commandArgs(trailingOnly=TRUE)

experimentName <- args[1]
csvPath <- args[2]
outputDir <- args[3]

colors = c('red', 'blue', 'grey', 'yellow', 'aquamarine', 'black', 'orange', 'cyan', 'violet', 'blueviolet', 'bisque4')

df <- read.csv(csvPath, header=TRUE)

#Rearrange groups left to right by plate layout
df$group <- factor(df$group, levels=as.character(unique(df[order(df$column),]$group)))

#Appends the sample size (n) to each group
ballislife<-levels(factor(df$group))
sample_size<-as.character(summary(df$group))
groups_with_n<-paste0(ballislife," (n=",sample_size,")")

coxfit <- coxph(Surv(last_time, as.logical(censored)) ~ strata(group), data=df)
capture.output(summary(coxfit), file = file.path(outputDir, "cox_hazard.txt"))

#Log-rank test
lrtest <- survdiff(Surv(last_time, as.logical(censored)) ~ group, data=df, rho=0)

#Plot and output Cox hazard results
pdf(file=file.path(outputDir, "hazard_plot.pdf"), width=10)
title = paste(experimentName, " Cumulative Hazards")
plot(survfit(coxfit), fun="cumhaz", main=title, xlab="Time (hr)",
     ylab="Cumulative risk of death", col=colors, lwd=5)
legend("topleft", legend=groups_with_n, col=colors, lwd=2, cex=.8)
