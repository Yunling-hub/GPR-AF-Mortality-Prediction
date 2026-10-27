# ============================================================
# ML preprocessing: parameters fitted on the training set and
# applied to the internal validation set and the NWICU cohort
# ============================================================

library(dplyr)
library(mice)

# ----- paths -----
train_path <- "data/splits/train_set.csv"
val_path   <- "data/splits/val_set.csv"
nwicu_path <- "data/raw/NWICU/df_nwicu.csv"

out_dir   <- "data/processed/ML"
param_dir <- "data/processed/ML/params"
dir.create(out_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(param_dir, recursive = TRUE, showWarnings = FALSE)

# ----- read data -----
train <- read.csv(train_path)
val   <- read.csv(val_path)
nwicu <- read.csv(nwicu_path)

cat("Training set:", nrow(train), "rows\n")
cat("Validation  :", nrow(val),   "rows\n")
cat("NWICU       :", nrow(nwicu), "rows\n")

# ============================================================
# 1. Extreme values (fixed thresholds, identical across cohorts)
# ============================================================
remove_outliers <- function(df) {
  df$Serum_glucose[df$Serum_glucose < 1 | df$Serum_glucose > 50] <- NA
  df$Serum_potassium[df$Serum_potassium < 1 | df$Serum_potassium > 10] <- NA
  
  df$WBC[df$WBC < 0.1 | df$WBC > 300] <- NA
  df$RBC[df$RBC < 0.5 | df$RBC > 10] <- NA
  df$Platelets[df$Platelets < 1 | df$Platelets > 2000] <- NA
  df$Hemoglobin[df$Hemoglobin < 1 | df$Hemoglobin > 30] <- NA
  df$RDW[df$RDW < 5 | df$RDW > 50] <- NA
  df$Hematocrit[df$Hematocrit < 5 | df$Hematocrit > 80] <- NA
  df$Serum_sodium[df$Serum_sodium < 90 | df$Serum_sodium > 180] <- NA
  df$Total_calcium[df$Total_calcium < 5 | df$Total_calcium > 14] <- NA
  df$Serum_chloride[df$Serum_chloride < 50 | df$Serum_chloride > 150] <- NA
  df$Anion_gap[df$Anion_gap < 0 | df$Anion_gap > 50] <- NA
  df$PT[df$PT < 5 | df$PT > 200] <- NA
  df$APTT[df$APTT < 5 | df$APTT > 200] <- NA
  df$BUN[df$BUN < 1 | df$BUN > 150] <- NA
  df$Serum_creatinine[df$Serum_creatinine < 0.1 | df$Serum_creatinine > 15] <- NA
  
  df$HR[df$HR < 10 | df$HR > 300] <- NA
  df$SBP[df$SBP < 40 | df$SBP > 300] <- NA
  df$DBP[df$DBP < 10 | df$DBP > 200] <- NA
  df$RR[df$RR < 0 | df$RR > 80] <- NA
  df$Spo2[df$Spo2 < 50 | df$Spo2 > 100] <- NA
  df$Temperature[df$Temperature < 28 | df$Temperature > 44] <- NA
  
  df
}

train <- remove_outliers(train)
val   <- remove_outliers(val)
nwicu <- remove_outliers(nwicu)

# ============================================================
# 2. Winsorization: quantiles from the training set, then
#    applied unchanged to val and NWICU
# ============================================================
to_winsor <- c("WBC","RBC","Platelets","Hemoglobin","RDW","Hematocrit",
               "Serum_sodium","Total_calcium","Serum_chloride","Anion_gap",
               "PT","APTT","BUN","Serum_creatinine",
               "HR","SBP","DBP","RR","Spo2","Temperature")

winsor_params <- list()

for (v in to_winsor) {
  if (v %in% names(train)) {
    q1  <- quantile(train[[v]], 0.01, na.rm = TRUE)
    q99 <- quantile(train[[v]], 0.99, na.rm = TRUE)
    winsor_params[[v]] <- c(q1 = q1, q99 = q99)
    
    train[[v]][train[[v]] < q1]  <- q1
    train[[v]][train[[v]] > q99] <- q99
    
    if (v %in% names(val)) {
      val[[v]][val[[v]] < q1]   <- q1
      val[[v]][val[[v]] > q99]  <- q99
    }
    if (v %in% names(nwicu)) {
      nwicu[[v]][nwicu[[v]] < q1]  <- q1
      nwicu[[v]][nwicu[[v]] > q99] <- q99
    }
  }
}

saveRDS(winsor_params, file.path(param_dir, "winsor_params.rds"))

# ============================================================
# 3. MICE: fitted on the training set, then applied to val and NWICU
# ============================================================
vars <- c(to_winsor, "Serum_glucose", "Serum_potassium")
vars <- vars[vars %in% names(train)]
vars <- vars[vars %in% names(val)]
vars <- vars[vars %in% names(nwicu)]

miss <- sapply(train[, vars], function(x) sum(is.na(x)))
miss <- miss[miss > 0]

if (length(miss) > 0) {
  v_miss <- names(miss)
  v_keep <- v_miss[miss / nrow(train) < 0.2]
  v_keep <- unique(c(v_keep, "Serum_glucose", "Serum_potassium"))
  v_keep <- v_keep[v_keep %in% names(train)]
  v_keep <- v_keep[v_keep %in% names(val)]
  v_keep <- v_keep[v_keep %in% names(nwicu)]
  
  set.seed(42)
  imp_train <- mice(train[, v_keep], m = 5, method = "pmm",
                    maxit = 5, printFlag = FALSE, ridge = 0.001)
  saveRDS(imp_train, file.path(param_dir, "mice_train.rds"))
  
  train[, v_keep] <- complete(imp_train, 1)
  
  imp_val   <- mice.mids(imp_train, newdata = val[, v_keep],   maxit = 0)
  imp_nwicu <- mice.mids(imp_train, newdata = nwicu[, v_keep], maxit = 0)
  
  val[, v_keep]   <- complete(imp_val, 1)
  nwicu[, v_keep] <- complete(imp_nwicu, 1)
} else {
  v_keep <- vars
}

# ============================================================
# 4. Median fallback: computed on the training set, applied to all three
# ============================================================
medians <- sapply(train[, vars], median, na.rm = TRUE)
saveRDS(medians, file.path(param_dir, "medians.rds"))

for (v in vars) {
  if (any(is.na(train[[v]]))) train[[v]][is.na(train[[v]])] <- medians[v]
  if (any(is.na(val[[v]])))   val[[v]][is.na(val[[v]])]     <- medians[v]
  if (any(is.na(nwicu[[v]]))) nwicu[[v]][is.na(nwicu[[v]])] <- medians[v]
}

# ============================================================
# 5. Gender coding, outcomes and categorical variables to numeric
# ============================================================
encode_common <- function(df) {
  if ("Gender" %in% names(df)) {
    if (is.character(df$Gender) | is.factor(df$Gender)) {
      df$Gender <- ifelse(df$Gender %in% c("M","Male","男","男性"), 0, 1)
    }
  }
  
  out_vars <- c("death_within_icu_30days","death_within_icu_90days","death_within_icu_365days")
  for (v in out_vars) {
    if (v %in% names(df)) df[[v]] <- as.numeric(df[[v]])
  }
  
  cat_vars <- c("Gender","Hyperlipidemia","Hypertension","Diabetes","HF","MI","MT",
                "CKD","ARF","Stroke","Sepsis","Vasopressor","Ventilation","CRRT")
  for (v in cat_vars) {
    if (v %in% names(df)) df[[v]] <- as.numeric(df[[v]])
  }
  
  df
}

train <- encode_common(train)
val   <- encode_common(val)
nwicu <- encode_common(nwicu)

# ============================================================
# 6. Save
# ============================================================
write.csv(train, file.path(out_dir, "train_processed.csv"), row.names = FALSE)
write.csv(val,   file.path(out_dir, "val_processed.csv"),   row.names = FALSE)
write.csv(nwicu, file.path(out_dir, "nwicu_processed.csv"), row.names = FALSE)

saveRDS(train, file.path(out_dir, "train_processed.rds"))
saveRDS(val,   file.path(out_dir, "val_processed.rds"))
saveRDS(nwicu, file.path(out_dir, "nwicu_processed.rds"))

cat("\nDone.\n")
cat("Training set:", nrow(train), "rows\n")
cat("Validation  :", nrow(val),   "rows\n")
cat("NWICU       :", nrow(nwicu), "rows\n")

cat("Expected: train ~11113 (80% of 13891), val ~2778 (20% of 13891)\n")

cat("Remaining missing values (train):\n"); print(colSums(is.na(train[, vars])))
cat("Remaining missing values (val):\n");   print(colSums(is.na(val[, vars])))
cat("Remaining missing values (NWICU):\n"); print(colSums(is.na(nwicu[, vars])))
cat("Missing 30-day outcome (train / val / NWICU):\n")
print(c(sum(is.na(train$death_within_icu_30days)),
        sum(is.na(val$death_within_icu_30days)),
        sum(is.na(nwicu$death_within_icu_30days))))