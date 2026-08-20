library(rms)
library(survival)

# ----- read data -----
train <- read.csv("./data/splits/train_set.csv")

# prepare survival data
train$time <- train$icu_survival_day_365
train$status <- train$death_within_icu_365days

# 7 predictors
features <- c("Age", "BUN", "GPR", "RDW", "Sepsis", "RR", "Platelets")

# fill missing (should be none after preprocessing)
for(feat in features) {
  if(any(is.na(train[, feat]))) {
    train[, feat][is.na(train[, feat])] <- median(train[, feat], na.rm = TRUE)
  }
}

train$Sepsis <- as.factor(train$Sepsis)

# ----- fit Cox model -----
dd <- datadist(train)
options(datadist = "dd")

cox_model <- cph(Surv(time, status) ~ Age + BUN + GPR + RDW + Sepsis + RR + Platelets,
                 data = train, x = TRUE, y = TRUE, surv = TRUE)

# baseline survival at 30, 90, 365 days
base_surv <- summary(survfit(cox_model), times = c(30, 90, 365))$surv

# probability functions
fun_30d <- function(x) return(1 - base_surv[1]^exp(x))
fun_90d <- function(x) return(1 - base_surv[2]^exp(x))
fun_365d <- function(x) return(1 - base_surv[3]^exp(x))

# ----- plot nomogram -----
tiff("./results/Nomogram.tiff", width = 5000, height = 3800, res = 600, compression = "lzw")

par(mar = c(5, 5, 4, 3))

nom <- nomogram(cox_model,
                fun = list(
                  `30-day Mortality` = fun_30d,
                  `90-day Mortality` = fun_90d,
                  `365-day Mortality` = fun_365d
                ),
                fun.at = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
                funlabel = c("30-day Mortality", "90-day Mortality", "365-day Mortality"),
                lp = FALSE,
                maxscale = 100,
                vnames = "labels")

plot(nom,
     xfrac = 0.15,
     lmgp = 0.20,
     cex.axis = 0.7,
     cex.var = 0.7,
     vgap = 0.5,
     col.grid = gray(0.95))

dev.off()

cat("Nomogram saved: ./results/Nomogram.tiff\n")