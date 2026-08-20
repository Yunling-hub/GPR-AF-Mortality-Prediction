library(dplyr)
library(mice)

# ----- load data -----
df <- mimic_raw_data
cat("Original data:", nrow(df), "rows x", ncol(df), "cols\n")

# winsorization function
winsor <- function(x) {
  q1 <- quantile(x, 0.01, na.rm=T)
  q99 <- quantile(x, 0.99, na.rm=T)
  x[x<q1] <- q1
  x[x>q99] <- q99
  x
}

# ----- remove outliers -----
df1 <- df

# glucose & potassium: only extreme values removed
df1$Serum_glucose[df1$Serum_glucose < 1 | df1$Serum_glucose > 50] <- NA
df1$Serum_potassium[df1$Serum_potassium < 1 | df1$Serum_potassium > 10] <- NA

# labs
df1$WBC[df1$WBC < 0.1 | df1$WBC > 300] <- NA
df1$RBC[df1$RBC < 0.5 | df1$RBC > 10] <- NA
df1$Platelets[df1$Platelets < 1 | df1$Platelets > 2000] <- NA
df1$Hemoglobin[df1$Hemoglobin < 1 | df1$Hemoglobin > 30] <- NA
df1$RDW[df1$RDW < 5 | df1$RDW > 50] <- NA
df1$Hematocrit[df1$Hematocrit < 5 | df1$Hematocrit > 80] <- NA
df1$Serum_sodium[df1$Serum_sodium < 90 | df1$Serum_sodium > 180] <- NA
df1$Total_calcium[df1$Total_calcium < 5 | df1$Total_calcium > 14] <- NA
df1$Serum_chloride[df1$Serum_chloride < 50 | df1$Serum_chloride > 150] <- NA
df1$Anion_gap[df1$Anion_gap < 0 | df1$Anion_gap > 50] <- NA
df1$PT[df1$PT < 5 | df1$PT > 200] <- NA
df1$APTT[df1$APTT < 5 | df1$APTT > 200] <- NA
df1$BUN[df1$BUN < 1 | df1$BUN > 150] <- NA
df1$Serum_creatinine[df1$Serum_creatinine < 0.1 | df1$Serum_creatinine > 15] <- NA

# vitals
df1$HR[df1$HR < 10 | df1$HR > 300] <- NA
df1$SBP[df1$SBP < 40 | df1$SBP > 300] <- NA
df1$DBP[df1$DBP < 10 | df1$DBP > 200] <- NA
df1$Mean_BP[df1$Mean_BP < 20 | df1$Mean_BP > 250] <- NA
df1$RR[df1$RR < 0 | df1$RR > 80] <- NA
df1$Spo2[df1$Spo2 < 50 | df1$Spo2 > 100] <- NA
df1$Temperature[df1$Temperature < 28 | df1$Temperature > 44] <- NA

# ----- winsorize (not for glucose & potassium) -----
to_winsor <- c("WBC","RBC","Platelets","Hemoglobin","RDW","Hematocrit",
               "Serum_sodium","Total_calcium","Serum_chloride","Anion_gap",
               "PT","APTT","BUN","Serum_creatinine",
               "HR","SBP","DBP","Mean_BP","RR","Spo2","Temperature")

set.seed(42)

for (v in to_winsor) {
  if (v %in% names(df1)) {
    n <- sum(!is.na(df1[[v]]))
    if (n > 50) {
      df1[[v]] <- winsor(df1[[v]])
    }
  }
}

# ----- check missing -----
vars <- c(to_winsor, "Serum_glucose", "Serum_potassium")
miss <- sapply(df1[,vars], function(x) sum(is.na(x)))
miss <- miss[miss > 0]
print(miss)

# ----- MICE imputation -----
if (length(miss) > 0) {
  v_miss <- names(miss)
  v_keep <- v_miss[miss / nrow(df1) < 0.2]
  v_keep <- unique(c(v_keep, "Serum_glucose", "Serum_potassium"))
  
  set.seed(42)
  imp <- mice(df1[,v_keep], m=5, method="pmm", maxit=5, printFlag=F, ridge=0.001)
  df_imp <- complete(imp, 1)
  df1[,v_keep] <- df_imp
}

df_final <- df1

# double-check missing
for (v in vars) {
  if (any(is.na(df_final[[v]]))) {
    df_final[[v]][is.na(df_final[[v]])] <- median(df_final[[v]], na.rm=T)
  }
}

# ----- GPR -----
df_final$GPR <- df_final$Serum_glucose / df_final$Serum_potassium

# ----- gender: male=0, female=1 -----
if ("Gender" %in% names(df_final)) {
  if (is.character(df_final$Gender) | is.factor(df_final$Gender)) {
    df_final$Gender <- ifelse(df_final$Gender %in% c("M","Male","男","男性"), 0, 1)
  }
}

# outcomes & categoricals to numeric
out_vars <- c("death_within_icu_30days","death_within_icu_90days","death_within_icu_365days")
for (v in out_vars) {
  if (v %in% names(df_final)) df_final[[v]] <- as.numeric(df_final[[v]])
}

cat_vars <- c("Gender","Hyperlipidemia","Hypertension","Diabetes","HF","MI","MT",
              "CKD","ARF","Stroke","Sepsis","Vasopressor","Ventilation","CRRT")
for (v in cat_vars) {
  if (v %in% names(df_final)) df_final[[v]] <- as.numeric(df_final[[v]])
}

# ----- save -----
dir.create("./data/processed/MIMIC/", recursive=T, showWarnings=F)
write.csv(df_final, "./data/processed/MIMIC/df_mimic_cleaned.csv", row.names=F)
saveRDS(df_final, "./data/processed/MIMIC/df_mimic_cleaned.rds")

cat("Done.", nrow(df_final), "rows x", ncol(df_final), "cols\n")