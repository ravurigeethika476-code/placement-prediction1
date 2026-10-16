# ==============================================================================
# SGBOOST_MODEL.R: STOCHASTIC GRADIENT BOOSTING IN R
# Student Placement Prediction Project
# ==============================================================================

# Install required packages if not already present
required_packages <- c("gbm", "caret", "ggplot2", "pROC")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) {
  cat("[+] Installing missing R packages:", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages, repos = "https://cloud.r-project.org")
}

library(gbm)
library(caret)
library(ggplot2)
library(pROC)

cat("==============================================================================\n")
cat("STOCHASTIC GRADIENT BOOSTING (SGBOOST) MODEL IN R\n")
cat("==============================================================================\n")

# 1. LOAD DATASET
data_file <- "student_placement_prediction_dataset_2026.csv"
if (!file.exists(data_file)) {
  if (file.exists("uploads/student_placement_prediction_dataset_2026.csv")) {
    data_file <- "uploads/student_placement_prediction_dataset_2026.csv"
  } else {
    stop("Error: Dataset file not found.")
  }
}

cat("[+] Loading dataset from:", data_file, "\n")
df <- read.csv(data_file, stringsAsFactors = FALSE)
cat("[+] Dataset Dimensions:", nrow(df), "rows x", ncol(df), "columns\n\n")

# 2. DATA PREPROCESSING
# Select relevant features
features <- c("cgpa", "coding_skill_score", "aptitude_score",
              "communication_skill_score", "internships_count",
              "projects_count", "backlogs", "mock_interview_score",
              "branch", "college_tier", "placement_status")

available_features <- intersect(features, colnames(df))
model_df <- na.omit(df[, available_features])

# Convert categorical features to factors
if ("branch" %in% colnames(model_df)) {
  model_df$branch <- as.factor(model_df$branch)
}
if ("college_tier" %in% colnames(model_df)) {
  model_df$college_tier <- as.factor(model_df$college_tier)
}

# Encode target variable: Placed = 1, Not Placed = 0 (bernoulli distribution requirement)
model_df$placement_target <- ifelse(model_df$placement_status == "Placed", 1, 0)
model_df$placement_status <- NULL

# 3. TRAIN / TEST SPLIT (80% Train, 20% Test)
set.seed(42)
train_indices <- sample(1:nrow(model_df), size = 0.8 * nrow(model_df))
train_data <- model_df[train_indices, ]
test_data  <- model_df[-train_indices, ]

cat("[+] Training set size :", nrow(train_data), "\n")
cat("[+] Testing set size  :", nrow(test_data), "\n\n")

# 4. FIT STOCHASTIC GRADIENT BOOSTING MODEL
# Note: bag.fraction = 0.8 implements stochastic gradient boosting (Friedman 2002)
cat("[+] Training Stochastic Gradient Boosting Model (gbm)...\n")
sgboost_model <- gbm(
  formula           = placement_target ~ .,
  data              = train_data,
  distribution      = "bernoulli",
  n.trees           = 150,
  interaction.depth = 4,
  shrinkage         = 0.08,
  bag.fraction      = 0.80,    # Stochastic subsampling per tree
  cv.folds          = 5,
  n.minobsinnode    = 10,
  verbose           = FALSE
)

# Determine optimal number of boosting iterations using 5-fold CV
best_trees <- gbm.perf(sgboost_model, method = "cv", plot.it = FALSE)
cat("[+] Optimal Boosting Trees (via 5-fold CV):", best_trees, "\n\n")

# 5. VARIABLE IMPORTANCE
cat("--- RELATIVE VARIABLE INFLUENCE ---\n")
var_importance <- summary(sgboost_model, n.trees = best_trees, plotit = FALSE)
print(head(var_importance, 10))

# 6. TEST SET PREDICTIONS & EVALUATION
pred_probs <- predict(sgboost_model, newdata = test_data, n.trees = best_trees, type = "response")
pred_classes <- ifelse(pred_probs >= 0.5, 1, 0)

actuals <- test_data$placement_target
cm <- table(Actual = actuals, Predicted = pred_classes)
cat("\n--- CONFUSION MATRIX ---\n")
print(cm)

accuracy  <- sum(diag(cm)) / sum(cm)
precision <- cm["1", "1"] / sum(cm[, "1"])
recall    <- cm["1", "1"] / sum(cm["1", ])
f1_score  <- 2 * (precision * recall) / (precision + recall)
roc_obj   <- roc(actuals, pred_probs)
auc_val   <- auc(roc_obj)

cat("\n--- MODEL PERFORMANCE METRICS ---\n")
cat(sprintf("Accuracy  : %.2f%%\n", accuracy * 100))
cat(sprintf("Precision : %.2f%%\n", precision * 100))
cat(sprintf("Recall    : %.2f%%\n", recall * 100))
cat(sprintf("F1 Score  : %.4f\n", f1_score))
cat(sprintf("ROC-AUC   : %.4f\n\n", auc_val))

# 7. PLOT VARIABLE IMPORTANCE
importance_df <- as.data.frame(var_importance)
colnames(importance_df) <- c("Variable", "Relative_Influence")

p <- ggplot(importance_df[1:min(8, nrow(importance_df)), ], 
            aes(x = reorder(Variable, Relative_Influence), y = Relative_Influence)) +
  geom_bar(stat = "identity", fill = "#3498db", color = "black") +
  coord_flip() +
  labs(title = "SGBoost: Relative Feature Importance",
       x = "Feature", y = "Relative Influence (%)") +
  theme_minimal()

print(p)
cat("[+] SGBoost Model training and evaluation in R completed successfully!\n")
