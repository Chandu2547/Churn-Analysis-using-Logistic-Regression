
# Customer Churn Analysis using Logistic Regression

## Project Overview

This project performs customer churn prediction using Logistic Regression in R.  
It identifies factors influencing customer attrition and builds a predictive model to classify customers as Stayed or Churned.

#Objective:  
To understand drivers of customer churn and develop a reliable model that predicts which customers are likely to leave a service.

---

## Dataset

- File: `Thesis Data.csv`
- Records represent customer demographics, service usage, billing, and account information.
- Target variable: `Customer_Status` (Stayed / Churned)

---

## Workflow Summary

1. Data Loading and Inspection  
   - Import dataset using `read.csv()`  
   - View structure, summary statistics, and initial rows using:
     ```r
     dim(churn)
     str(churn)
     summary(churn)
     head(churn)
     ```

2. Data Cleaning and Transformation  
   - Convert numeric columns to numeric and categorical variables to factors:
     ```r
     num_cols <- c("Age","Number_of_Referrals","Tenure_in_Months", "Monthly_Charge", "Total_Charges", ...)
     churn[num_cols] <- lapply(churn[num_cols], as.numeric)
     ```
   - Convert `Customer_Status` into binary factor (`Stayed`, `Churned`)

3. Exploratory Data Analysis (EDA)  
   Used ggplot2 for data visualization:
   - Churn distribution  
   - Churn by contract type  
   - Churn vs tenure  
   - Monthly charge vs churn  

   ```r
   ggplot(churn, aes(x = Customer_Status, fill = Customer_Status)) +
     geom_bar() + scale_fill_manual(values = c("red", "green")) +
     labs(title = "Churn Distribution", y = "Count")
````

4. **Feature Engineering**

   * Created `tenure_group` to categorize customers by subscription length:

     ```r
     churn$tenure_group <- cut(churn$Tenure_in_Months,
                               breaks=c(-Inf,6,12,24,48,Inf),
                               labels=c("0-6","7-12","13-24","25-48","49+"))
     ```
   * Calculated `avg_charge_per_month`:

     ```r
     churn$avg_charge_per_month <- ifelse(churn$Tenure_in_Months > 0,
                                          churn$Total_Charges / churn$Tenure_in_Months,
                                          churn$Monthly_Charge)
     ```

5. **Data Splitting**

   * Split dataset into training (70%) and testing (30%) using `caTools`:

     ```r
     library(caTools)
     set.seed(123)
     split <- sample.split(churn$Customer_Status, SplitRatio = 0.7)
     train_data <- subset(churn, split == TRUE)
     test_data  <- subset(churn, split == FALSE)
     ```

6. **Model Training (Logistic Regression)**

   * Used **caret** for model training and cross-validation:

     ```r
     library(caret)
     churn_ctrl <- trainControl(method = "cv", number = 5,
                                classProbs = TRUE,
                                summaryFunction = twoClassSummary)

     glm_model <- train(Customer_Status ~ . - Customer_ID,
                        data = train_data,
                        method = "glm",
                        family = "binomial",
                        trControl = churn_ctrl,
                        metric = "ROC")
     ```
   * Identified significant predictors and built a refined model using only key variables:

     ```r
     new_model <- train(Customer_Status ~ Contract + Online_Security + 
                        Payment_Method + Monthly_Charge + Total_Charges + 
                        Paperless_Billing + Gender + Premium_Support + Internet_Type,
                        data = train_data,
                        method = "glm",
                        family = "binomial",
                        trControl = churn_ctrl,
                        metric = "ROC")
     ```

7. **Model Evaluation**

   * Predicted churn probabilities and labels:

     ```r
     glm_probs <- predict(new_model, newdata = test_data, type = "prob")[, "Churned"]
     glm_preds <- factor(ifelse(glm_probs > 0.5, "Churned", "Stayed"),
                         levels = c("Stayed","Churned"))
     ```
   * Evaluated model using **confusion matrix** and **ROC curve**:

     ```r
     CM <- confusionMatrix(glm_preds, test_data$Customer_Status, positive = "Churned")
     print(CM)

     library(pROC)
     roc_obj <- roc(test_data$Customer_Status, glm_probs, levels = c("Stayed","Churned"))
     plot(roc_obj, main = "ROC Curve - Logistic Regression")
     auc(roc_obj)
     ```

---

## Key Findings

| Metric      | Value | Description                                     |
| ----------- | ----- | ----------------------------------------------- |
| Accuracy    | 77.6% | Model correctly classifies ~78% of customers    |
| Sensitivity | 65.7% | Correctly detects ~66% of churned customers     |
| Specificity | 83.5% | Correctly identifies ~83% of retained customers |
| AUC         | ~0.85 | Strong discriminative performance               |

**Interpretation:**

* Longer contracts (1 or 2 years) significantly reduce churn risk.
* Online security and premium support services lower churn likelihood.
* Higher monthly charges and paperless billing slightly increase churn risk.
* Gender and Internet type also have measurable effects.

---

## Model Insights

The logistic regression model indicates:

* **Contract Type (One/Two Year):** Strongly reduces churn odds
* **Online Security & Premium Support:** Decrease churn risk significantly
* **Payment Method (Credit Card):** Associated with reduced churn
* **Monthly Charge:** Higher values increase churn probability
* **Total Charges:** Higher total spending reduces churn
* **Paperless Billing:** Slightly increases churn odds

These insights can guide retention strategies such as offering incentives for longer contracts and promoting add-on services that improve customer satisfaction.

---

## Visualizations

| Visualization                                         | Description                                                       |
| ----------------------------------------------------- | ----------------------------------------------------------------- |
| ![Churn Distribution]()        | Overall churn distribution by tenure                              |
| ![Contract Type](Churn%20by%20Contract%20Type.png)    | Churn rate across different contract types                        |
| ![Monthly Charges](Monthly%20Charge%20by%20Churn.png) | Monthly charges comparison between churned and retained customers |
| ![ROC Curve](ROC%20curve.png)                         | ROC curve illustrating model performance                          |

---

## Results Summary

| Metric            | Score  |
| ----------------- | ------ |
| Accuracy          | 77.61% |
| Sensitivity       | 65.75% |
| Specificity       | 83.48% |
| Kappa             | 0.49   |
| Balanced Accuracy | 74.62% |
| AUC               | ~0.85  |

The model demonstrates good balance between identifying churners and non-churners, with a strong AUC score, indicating effective discrimination.


```
