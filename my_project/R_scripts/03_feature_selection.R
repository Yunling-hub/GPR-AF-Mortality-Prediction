library(tidyverse)
library(glmnet)
library(Boruta)
library(corrplot)
library(caret)

# ----- read data -----
train_data <- read.csv("./data/splits/train_set.csv")

cat("Training set:", nrow(train_data), "rows,", ncol(train_data), "cols\n")

# ----- candidate variables -----
all_predictors <- c(
  "GPR", "Age", "Gender",
  "HR", "SBP", "DBP", "Mean_BP", "RR", "SpO2", "Temperature",
  "WBC", "RBC", "Platelets", "Hemoglobin", "RDW", "Hematocrit",
  "Serum_sodium", "Total_calcium", "PT", "APTT", "BUN", "Serum_creatinine",
  "Hyperlipidemia", "Hypertension", "Diabetes", "HF", "MI", "MT",
  "CKD", "ARF", "Stroke", "Sepsis",
  "Vasopressor", "Ventilation", "CRRT"
)

available <- intersect(all_predictors, names(train_data))
cat("Candidate variables:", length(available), "\n")

X <- train_data[available]
y <- train_data$death_within_icu_30days

# convert to numeric
if("Gender" %in% names(X)) X$Gender <- as.numeric(X$Gender)
X <- as.data.frame(lapply(X, as.numeric))

# fill missing (should be none after preprocessing)
for(i in 1:ncol(X)) {
  if(any(is.na(X[,i]))) {
    X[is.na(X[,i]), i] <- median(X[,i], na.rm = TRUE)
  }
}

# ----- remove collinearity (|r| > 0.7) -----
cat("\nRemoving collinear variables (|r| > 0.7)\n")

cor_all <- cor(X, use = "complete.obs")
high_cor <- which(abs(cor_all) > 0.7 & upper.tri(cor_all), arr.ind = TRUE)

if(nrow(high_cor) > 0) {
  cat("Highly correlated pairs found:\n")
  for(i in 1:nrow(high_cor)) {
    v1 <- rownames(cor_all)[high_cor[i, 1]]
    v2 <- colnames(cor_all)[high_cor[i, 2]]
    cat(sprintf("  %s -- %s: r = %.3f\n", v1, v2, cor_all[v1, v2]))
  }
  
  to_remove <- findCorrelation(cor_all, cutoff = 0.7, names = TRUE)
  if("GPR" %in% to_remove) {
    to_remove <- setdiff(to_remove, "GPR")
    cat("  GPR kept (decision left to algorithms)\n")
  }
  
  if(length(to_remove) > 0) {
    X <- X[, !(names(X) %in% to_remove)]
    available <- setdiff(available, to_remove)
    cat("  Removed:", paste(to_remove, collapse = ", "), "\n")
  }
}

X_mat <- as.matrix(X)

# remove zero-variance variables
zero_var <- which(apply(X_mat, 2, var) == 0)
if(length(zero_var) > 0) {
  X_mat <- X_mat[, -zero_var, drop = FALSE]
  available <- available[-zero_var]
}

# ----- LASSO regression (10-fold CV) -----
cat("\nLASSO regression (10-fold CV)\n")

set.seed(123)
lambda_grid <- 10^seq(-8, 1, length = 200)
cv_lasso <- cv.glmnet(X_mat, y, family = "binomial", alpha = 1,
                      nfolds = 10, lambda = lambda_grid)

coef_min <- coef(cv_lasso, s = "lambda.min")
coef_1se <- coef(cv_lasso, s = "lambda.1se")

lasso_vars <- rownames(coef_1se)[which(coef_1se[,1] != 0)]
lasso_vars <- setdiff(lasso_vars, "(Intercept)")

cat("LASSO selected (lambda.1se):", length(lasso_vars), "variables\n")
cat("GPR selected:", "GPR" %in% lasso_vars, "\n")

# save coefficients
coef_df <- data.frame(
  variable = rownames(coef_min),
  coef_min = as.vector(coef_min[,1]),
  coef_1se = as.vector(coef_1se[,1])
)
write.csv(coef_df, "./results/LASSO_coefficients.csv", row.names = FALSE)

# LASSO path plot
tiff("./results/LASSO_path.tiff", width = 8, height = 6, units = "in", res = 600, compression = "lzw")
par(mar = c(5,5,4,1))
log_lambda <- log(cv_lasso$lambda)
coef_mat <- as.matrix(coef(cv_lasso$glmnet.fit)[-1,])
plot(log_lambda, rep(0, length(log_lambda)), type = "n",
     xlab = "Log Lambda", ylab = "Coefficients",
     xlim = c(-9, -2.3), ylim = range(coef_mat))
for(i in 1:nrow(coef_mat)) {
  lines(log_lambda, coef_mat[i,], col = rainbow(nrow(coef_mat))[i], lwd = 1.2)
}
abline(v = log(cv_lasso$lambda.min), lty = 2, col = "black", lwd = 1.5)
abline(v = log(cv_lasso$lambda.1se), lty = 2, col = "gray50", lwd = 1.5)
legend("bottomright", legend = c("lambda.min", "lambda.1se"),
       col = c("black", "gray50"), lty = 2, lwd = 1.5)
dev.off()
cat("LASSO_path.tiff saved\n")

# CV error plot
tiff("./results/LASSO_cv.tiff", width = 8, height = 6, units = "in", res = 600, compression = "lzw")
par(mar = c(5,5,4,2))
idx <- log_lambda >= -9 & log_lambda <= -2.3
plot(log_lambda[idx], cv_lasso$cvm[idx], type = "b", pch = 19, cex = 0.8,
     col = "red", xlab = "Log Lambda", ylab = "Binomial Deviance",
     xlim = c(-9, -2.3), ylim = range(cv_lasso$cvm[idx] + cv_lasso$cvsd[idx]))
arrows(log_lambda[idx], cv_lasso$cvm[idx] - cv_lasso$cvsd[idx],
       log_lambda[idx], cv_lasso$cvm[idx] + cv_lasso$cvsd[idx],
       length = 0.02, angle = 90, code = 3, col = "grey50")
abline(v = log(cv_lasso$lambda.min), lty = 2, col = "black", lwd = 1.5)
abline(v = log(cv_lasso$lambda.1se), lty = 2, col = "gray50", lwd = 1.5)
legend("bottomright", legend = c("lambda.min", "lambda.1se"),
       col = c("black", "gray50"), lty = 2, lwd = 1.5)
dev.off()
cat("LASSO_cv.tiff saved\n")

# ----- Boruta algorithm -----
cat("\nBoruta algorithm (may take a few minutes)\n")

set.seed(123)
boruta_data <- cbind(X, y = as.factor(y))
boruta_result <- Boruta(y ~ ., data = boruta_data, doTrace = 0, maxRuns = 100)
boruta_final <- TentativeRoughFix(boruta_result)
boruta_confirmed <- getSelectedAttributes(boruta_final, withTentative = FALSE)

cat("Boruta confirmed:", length(boruta_confirmed), "variables\n")
cat("GPR confirmed:", "GPR" %in% boruta_confirmed, "\n")

# Boruta importance plot
tiff("./results/Boruta_importance.tiff", width = 12, height = 8, units = "in", res = 300, compression = "lzw")
plot(boruta_result, las = 2, cex.axis = 0.7)
dev.off()
cat("Boruta_importance.tiff saved\n")

boruta_imp <- attStats(boruta_result)
boruta_imp <- boruta_imp[order(boruta_imp$meanImp, decreasing = TRUE),]
write.csv(boruta_imp, "./results/Boruta_importance.csv")

# ----- intersection of LASSO and Boruta -----
cat("\nIntersection of LASSO and Boruta\n")

final_vars <- intersect(lasso_vars, boruta_confirmed)

if(length(final_vars) < 5) {
  cat("  Few variables in intersection, adding top Boruta variables\n")
  top_boruta <- rownames(boruta_imp)[boruta_imp$decision == "Confirmed"]
  final_vars <- unique(c(final_vars, top_boruta[1:10]))
}

cat("Final variables:", length(final_vars), "\n")
cat(paste(final_vars, collapse = ", "), "\n")

# ----- re-check collinearity (|r| > 0.7) -----
cat("\nRe-checking collinearity (|r| > 0.7)\n")

if(length(final_vars) > 1) {
  data_cor <- train_data[, final_vars, drop = FALSE]
  data_cor <- as.data.frame(lapply(data_cor, function(x) {
    if(is.factor(x) || is.character(x)) as.numeric(as.factor(x)) else as.numeric(x)
  }))
  
  cor_final <- cor(data_cor, use = "complete.obs")
  high_cor_final <- which(abs(cor_final) > 0.7 & upper.tri(cor_final), arr.ind = TRUE)
  
  if(nrow(high_cor_final) > 0) {
    cat("  Highly correlated pairs found:\n")
    for(i in 1:nrow(high_cor_final)) {
      v1 <- rownames(cor_final)[high_cor_final[i, 1]]
      v2 <- colnames(cor_final)[high_cor_final[i, 2]]
      cat(sprintf("    %s -- %s: r = %.3f\n", v1, v2, cor_final[v1, v2]))
    }
    to_remove <- findCorrelation(cor_final, cutoff = 0.7, names = TRUE)
    if("GPR" %in% to_remove) to_remove <- setdiff(to_remove, "GPR")
    if(length(to_remove) > 0) {
      final_vars <- setdiff(final_vars, to_remove)
      cat("  Removed:", paste(to_remove, collapse = ", "), "\n")
    }
  }
}

write.csv(data.frame(variable = final_vars), "./results/final_predictors.csv", row.names = FALSE)

# ----- correlation heatmap -----
cat("\nCorrelation heatmap\n")

if(length(final_vars) > 1) {
  plot_data <- train_data[, final_vars, drop = FALSE]
  plot_data <- as.data.frame(lapply(plot_data, function(x) {
    if(is.factor(x) || is.character(x)) as.numeric(as.factor(x)) else as.numeric(x)
  }))
  
  cor_plot <- cor(plot_data, use = "complete.obs")
  
  tiff("./results/Correlation_heatmap.tiff", width = 12, height = 12,
       units = "in", res = 300, compression = "lzw")
  corrplot(cor_plot, method = "color", type = "full", order = "hclust",
           tl.col = "black", tl.cex = 0.8, tl.srt = 45,
           col = colorRampPalette(c("blue", "white", "red"))(200))
  dev.off()
  cat("Correlation_heatmap.tiff saved\n")
}

# ----- save selected dataset -----
vars_keep <- c("subject_id", final_vars,
               "death_within_icu_30days", "death_within_icu_90days", "death_within_icu_365days")
vars_keep <- intersect(vars_keep, names(train_data))
train_selected <- train_data[, vars_keep]
write.csv(train_selected, "./results/train_selected.csv", row.names = FALSE)

# ----- summary -----
cat("\n========== Summary ==========\n")
cat("Collinearity threshold: 0.7\n")
cat("Candidate variables:", length(available), "\n")
cat("LASSO selected (lambda.1se):", length(lasso_vars), "\n")
cat("Boruta confirmed:", length(boruta_confirmed), "\n")
cat("Final variables:", length(final_vars), "\n")
cat("\nFinal variable list:\n")
cat(paste(final_vars, collapse = ", "), "\n")
cat("\nGPR in final set:", "GPR" %in% final_vars, "\n")
cat("\nDone.\n")