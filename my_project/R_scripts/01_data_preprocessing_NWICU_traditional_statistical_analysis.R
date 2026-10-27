library(dplyr)
library(mice)

# ----- load data -----
df <- nwicu_raw_data
cat("Original:", nrow(df), "rows\n")

# winsorization function
winsor <- function(x) {
  q1 <- quantile(x, 0.01, na.rm = T)
  q99 <- quantile(x, 0.99, na.rm = T)
  x[x < q1] <- q1
  x[x > q99] <- q99
  x
}

# ----- outliers -----

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

# NWICU doesn't have Mean_BP
df$HR[df$HR < 10 | df$HR > 300] <- NA
df$SBP[df$SBP < 40 | df$SBP > 300] <- NA
df$DBP[df$DBP < 10 | df$DBP > 200] <- NA
df$RR[df$RR < 0 | df$RR > 80] <- NA
df$SpO2[df$SpO2 < 50 | df$SpO2 > 100] <- NA
df$Temperature[df$Temperature < 28 | df$Temperature > 44] <- NA

# ----- winsorize -----
to_winsor <- c("WBC","RBC","Platelets","Hemoglobin","RDW","Hematocrit",
               "Serum_sodium","Total_calcium","Serum_chloride","Anion_gap",
               "PT","APTT","BUN","Serum_creatinine",
               "HR","SBP","DBP","RR","SpO2","Temperature")  # no Mean_BP

set.seed(42)
for (v in to_winsor) {
  if (v %in% names(df)) {
    if (sum(!is.na(df[[v]])) > 50) df[[v]] <- winsor(df[[v]])
  }
}

# ----- check & impute -----
# Note: for traditional analyses, MICE is fitted separately within the
# MIMIC-IV and NWICU cohorts, so that imputed values reflect the
# distribution of each cohort. For ML analyses, imputation is fitted on
# the training set only (see the ML preprocessing script).
vars <- c(to_winsor, "Serum_glucose", "Serum_potassium")
miss <- sapply(df[, vars], function(x) sum(is.na(x)))
miss <- miss[miss > 0]
print(miss)

if (length(miss) > 0) {
  v_miss <- names(miss)
  v_keep <- v_miss[miss / nrow(df) < 0.2]
  v_keep <- unique(c(v_keep, "Serum_glucose", "Serum_potassium"))
  
  set.seed(42)
  imp <- mice(df[, v_keep], m = 5, method = "pmm", maxit = 5,
              printFlag = F, ridge = 0.001)
  df[, v_keep] <- complete(imp, 1)
}

df_final <- df

# final check
for (v in vars) {
  if (any(is.na(df_final[[v]]))) {
    df_final[[v]][is.na(df_final[[v]])] <- median(df_final[[v]], na.rm = T)
  }
}

# ----- GPR -----
df_final$GPR <- df_final$Serum_glucose / df_final$Serum_potassium

# gender
if ("Gender" %in% names(df_final)) {
  if (is.character(df_final$Gender) | is.factor(df_final$Gender)) {
    df_final$Gender <- ifelse(df_final$Gender %in% c("M", "Male", "男", "男性"), 0, 1)
  }
}

# outcomes
out_vars <- c("death_within_icu_30days", "death_within_icu_90days", "death_within_icu_365days")
for (v in out_vars) {
  if (v %in% names(df_final)) df_final[[v]] <- as.numeric(df_final[[v]])
}

cat_vars <- c("Gender","Hyperlipidemia","Hypertension","Diabetes","HF","MI","MT",
              "CKD","ARF","Stroke","Sepsis","Vasopressor","Ventilation","CRRT")
for (v in cat_vars) {
  if (v %in% names(df_final)) df_final[[v]] <- as.numeric(df_final[[v]])
}

# ----- save -----
out_dir <- "data/processed/NWICU"
dir.create(out_dir, recursive = T, showWarnings = F)

write.csv(df_final,
          file.path(out_dir, "df_nwicu_traditional_cleaned.csv"),
          row.names = F)

saveRDS(df_final,
        file.path(out_dir, "df_nwicu_traditional_cleaned.rds"))

cat("Done.", nrow(df_final), "rows\n")