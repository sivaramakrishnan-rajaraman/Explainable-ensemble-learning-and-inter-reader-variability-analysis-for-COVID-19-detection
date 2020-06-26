#restarting console
#Ctrl+Shift+F10 (Windows and Linux) 

#Reading csv file into R
my_data<- read.csv("./anova_data_cxr.csv")
head(my_data)

# Show a random sample
set.seed(1234)
dplyr::sample_n(my_data, 10)

# Compute summary statistics by groups - count, mean, sd:
#for mAP
library(dplyr)
group_by(my_data, model) %>%
  summarise(
    count = n(),
    mean = mean(map, na.rm = TRUE),
    sd = sd(map, na.rm = TRUE)
  )

#Visualize data
#we'll use the ggpubr R package for an easy ggplot2-based data visualization.
#Install the latest version of ggpubr from GitHub as follows (recommended):
#if(!require(devtools)) install.packages("devtools")
#devtools::install_github("kassambara/ggpubr")

#Visualize your data with ggpubr
# Box plots
# ++++++++++++++++++++
# Plot accuracy by model and color by group
library("ggpubr")
tiff("./ANOVA_R/box_plot1.tiff", units="in", width=5, height=5, res=400)
ggboxplot(my_data, x = "model", y = "map", res = 300,
          color = "model", palette = c("#00AFBC", "#E7B800", "#A6B800"),
          order = c("Rad-1", "Rad-2", "STAPLE"),
          ylab = "MAP", xlab = "Annotations")

dev.off()

# Mean plots: Accuracy
# ++++++++++++++++++++
# Plot weight by group
# Add error bars: mean_se
# (other values include: mean_sd, mean_ci, median_iqr, ....)
library("ggpubr")
ggline(my_data, x = "model", y = "map",
       add = c("mean_se", "jitter"), 
       order = c("Rad-1", "Rad-2", "STAPLE"),
       ylab = "MAP", xlab = "Annotations")

#If you still want to use R base graphs, type the following scripts:

# Box plot with 400 dpi
tiff("./ANOVA_R/boxplotr.tiff", units="in", width=5, height=5, res=400)
# insert ggplot code
boxplot(map ~ model, data = my_data, 
        ylab = "MAP", xlab = "Annotations",
        frame = FALSE, col = c("#00AFBB", "#E7B800", "#FC4E07", "#A6B800"))
dev.off()


# plotmeans
library("gplots")
tiff("./ANOVA_R/mean_plot.tiff", units="in", width=5, height=5, res=400)
plotmeans(map ~ model, data = my_data, frame = FALSE,
          ylab = "MAP", xlab = "Annotations",
          main="Mean Plot with 95% CI") 
dev.off()

#compute one way ANOVA
#We want to know if there is any significant difference between the mAP of two radiologists.
#The R function aov() can be used to answer to this question. 
#However, prior to using ANOVA, we need to see if certain assumptions are met.

#Check ANOVA assumptions: test validity?
#The ANOVA test assumes that, the data are normally distributed and the variance across groups are homogeneous. 
#We can check that with some diagnostic plots.

#Check the homogeneity of variance assumption
#The residuals versus fits plot can be used to check the homogeneity of variances.
# Compute the analysis of variance: MAP
aov_model1 <- aov(map ~ as.factor(model), data=my_data) 
# Summary of the analysis
summary.aov(aov_model1)
# Homogeneity of variances
plot(aov_model1, 1)

#In the plots, there is no evident relationships between residuals 
#and fitted values (the mean of each groups), which is good. 
#So, we can assume the homogeneity of variances.

# Normality
#for MAP
tiff("./ANOVA_R/Normal_Q_Q_plot.tiff", units="in", width=5, height=5, res=400)
plot(aov_model1, 2)
dev.off()

#As all the points fall approximately along this reference line, we can assume normality.

library(car)
#for accuracy
leveneTest(map ~ as.factor(model), data = my_data)

#From the output above we can see that the p-value is not less than the significance level of 0.05. 
#This means that there is no evidence to suggest that the variance across groups 
#is statistically significantly different. 
#Therefore, we can assume the homogeneity of variances in the different groups.
# so we can do one way anova

#Check the normality assumption for one-way ANOVA.
# Normality plot of residuals. 
#In the plot below, the quantiles of the residuals are plotted against the quantiles of the normal distribution. 
#A 45-degree reference line is also plotted.
# The normal probability plot of residuals is used to check the assumption 
#that the residuals are normally distributed. It should approximately follow a straight line.

# Extract the residuals: for MAP
aov_residuals_map <- residuals(object = aov_model1 )

# Run Shapiro-Wilk test
shapiro.test(x = aov_residuals_map )

#The conclusion above, is supported by the Shapiro-Wilk test on the ANOVA residuals 
#(W = 0.96614, p-value = 0.4395) for accuracy which finds no indication that normality is violated.

#The function summary.aov() is used to summarize the analysis of variance model.

# Compute the analysis of variance
aov_model1 <- aov(map ~ as.factor(model), data=my_data) 
# Summary of the analysis
summary.aov(aov_model1)

#The output includes the columns F value and Pr(>F) corresponding to the p-value of the test.

#Interpret the result of one-way ANOVA tests
#As the p-value is greater than the significance level 0.05, we can conclude that there are 
#no statistically significant differences between the groups highlighted with "*" in the model summary.