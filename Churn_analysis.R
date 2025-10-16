
"---------- CUSTOMER CHURN ANALYSIS ----------"

# SETTING WORKING DIRECTORY 

setwd("C:/Analytics/Practice datasets")

churn<-read.csv('Thesis Data.csv')

---# DATA INSPECTION (Checking dataset dimensions, structure, summary stats, and sample rows)

dim(churn)
str(churn)
summary(churn)
head(churn)

---# DATA CLEANING & Converting numeric columns to numeric type, categorical columns to factors.


num_cols <- c("Age","Number_of_Referrals","Tenure_in_Months",
              "Monthly_Charge","Total_Charges","Total_Refunds",
              "Total_Extra_Data_Charges","Total_Long_Distance_Charges",
              "Total_Revenue")

churn[num_cols] <- lapply(churn[num_cols], as.numeric)


cat_cols <- c("Gender","Married","State","Phone_Service","Multiple_Lines",
              "Internet_Service","Internet_Type","Online_Security",
              "Online_Backup","Device_Protection_Plan","Premium_Support",
              "Streaming_TV","Streaming_Movies","Streaming_Music",
              "Unlimited_Data","Contract","Paperless_Billing","Payment_Method")

churn[cat_cols] <- lapply(churn[cat_cols], as.factor)

table(is.na(churn))


---# PREPARING DATA FOR CHURN ANALYSIS 

unique(churn$Customer_Status)

#converting Customer_Status into a binary factor with levels "Stayed" and "Churned"

churn$Customer_Status <- ifelse(churn$Customer_Status == "Churned", "Churned", "Stayed")

churn$Customer_Status <- factor(churn$Customer_Status, levels = c("Stayed", "Churned"))

# Displaying counts, Displaying proportions using "table & prop.table"

table(churn$Customer_Status)

prop.table(table(churn$Customer_Status))


--- # EXPLORATORY DATA ANALYSIS


install.packages("ggplot2")
library(ggplot2)

  
# 1.Plotting overall churn distribution.
  
ggplot(churn, aes(x = Customer_Status, fill = Customer_Status)) +
  geom_bar() + scale_fill_manual(values = c("red", "green")) +
  labs(title = "Churn Distribution", y = "Count")

# 2.Churn by Contract Type
  
ggplot(churn, aes(x = Contract, fill = Customer_Status)) +
  geom_bar() + scale_fill_manual(values = c("red", "green"))
  labs(title="Churn Rate by Contract Type", y="Proportion")

# 3.Churn vs Tenure
  
  ggplot(churn, aes(x = Tenure_in_Months, fill = Customer_Status)) +
    geom_histogram(binwidth = 2, position = "fill") +
    scale_fill_manual(values = c("Stayed" = "red", "Churned" = "green")) +
    labs(title = "Churn by Tenure (Months)", y = "Proportion")
  
# 4.Monthly Charge by Churn

  ggplot(churn, aes(x = Customer_Status, y = Monthly_Charge, fill = Customer_Status)) +
    geom_boxplot() +scale_fill_manual(values = c("red", "green"))
    labs(title="Monthly Charges by Churn Status")
    
----- # Data Transformation for Predictor Engineering (tenure group & avg_charge_per_month)

      # Creating Tenure Groups
      
 churn$tenure_group <- cut(churn$Tenure_in_Months,
    breaks=c(-Inf,6,12,24,48,Inf),
      labels=c("0-6","7-12","13-24","25-48","49+"))

    
    # Calculating Average Charge per Month   
    
  churn$avg_charge_per_month <- ifelse(churn$Tenure_in_Months > 0,
      churn$Total_Charges / churn$Tenure_in_Months,
          churn$Monthly_Charge)

 
 # Converting the tenure_group variable into a factor (categorical variable)
 
churn$tenure_group <- as.factor(churn$tenure_group)

----- # DATA SPLITTING 
  
library(caTools)

set.seed(123)

split<-sample.split(churn$Customer_Status , SplitRatio = 0.7)

train_data <- subset(churn, split == TRUE)

test_data <- subset(churn, split == FALSE)

# Setting up cross-validation on the training data 


library(caret)

# IDENTIFYING COLUMNS WITH 1 VARIABLE,WE ONLY NEED COLUMNS WITH 2 VARIABLES 
   #BECAUSE OUR TARGET VARIABLE (customer_status) IS WITH 2 VARIABLES .

sapply(train_data, function(x) if (is.factor(x)) length(levels(x)) else NA)

#IDENTIFIED 2 COLUMNS WITH 1 VARIABLE AND REMOVED THEM FROM MODEL

train_data$Phone_Service <- NULL
train_data$Internet_Service <- NULL


 #cross-validation (CARET LIBRARY FOR trainControl)

install.packages("caret")

library(caret)

churn_ctrl <- trainControl(method = "cv", number = 5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

# GLM MODEL (Training logistic regression using all predictors except Customer_ID)

glm_model <- train(Customer_Status ~ . - Customer_ID,
                   data = train_data,
                   method = "glm",
                   family = "binomial",
                   trControl = churn_ctrl,
                   metric = "ROC")

summary(glm_model)

# Model shows several key variables significantly influence customer churn:
 # Significance levels: *** p < 0.001 , ** p < 0.01 , p < 0.05
  
# Contract Type (One Year, Two Year): Strongly reduces churn risk (***).

# Online Security (Yes): Significantly lowers churn likelihood (***).

# Payment Method (Credit Card): Associated with reduced churn (***).

# Monthly Charge: Higher charges increase churn risk (**).

# Total Charges: Higher total charges reduce churn risk (***).
 
# Paperless Billing (Yes): Slightly increases churn risk (**).

# Gender (Male): Males have lower churn odds (*).

# Premium Support (Yes): Reduces churn risk (*).

# Internet Type (DSL): Lowers churn likelihood (*).'


# training new model with cross-validation, 
   # using only the significant predictors identified from glm model

churn_ctrl <- trainControl(method = "cv", number = 5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

# Train new glm model using significant predictors only

new_model<- train(Customer_Status ~ Contract + Online_Security + 
                      Payment_Method + Monthly_Charge + Total_Charges + 
                      Paperless_Billing + Gender  + Premium_Support + Internet_Type,
                       data = train_data,
                       method = "glm",
                       family = "binomial",
                       trControl = churn_ctrl,
                       metric = "ROC")
summary(new_model)

"The new_model highlights contract length, online security, 
payment method, and pricing as key factors significantly affecting churn.
Longer contracts and services like online security and premium support reduce churn risk,
while higher monthly charges increase it."


-------# MODEL EVALUATION (Predicting probabilities) ----------

glm_probs<- predict(new_model, newdata = test_data, type = "prob")[, "Churned"]

# Converting to labels at 0.5 threshold

glm_preds <- factor(ifelse(glm_probs > 0.5, "Churned", "Stayed"),
                    levels = c("Stayed","Churned"))

table(glm_preds)

# CONFUSION MATRIX 

CM <-confusionMatrix(glm_preds, test_data$Customer_Status, positive = "Churned")

CM

"INTERPRETATION"

#Accuracy (77.61%): The model correctly predicts the customer’s status about 78% of the time, which is substantially better than the baseline no-information rate (66.87%).

#Sensitivity (65.75%): Of all actual churned customers, the model correctly identifies about 66%. This reflects the model’s ability to detect churners.
             
#Specificity (83.48%): Of all customers who stayed, about 83.5% are correctly predicted as non-churners, showing good performance on the majority class.
                          
#Kappa (0.4935): Indicates moderate agreement beyond chance between predicted and actual classifications.
                          
#Balanced Accuracy (74.62%): Averaging sensitivity and specificity, showing decent balanced performance across classes.
                          
 
# Computing ROC curve object using test labels and predicted probabilities

install.packages("pROC")  

library(pROC) 

roc_obj <- roc(test_data$Customer_Status, glm_probs, levels = c("Stayed","Churned"))

# Calculating  AUC value

auc_value <- auc(roc_obj)
print(auc_value)

## Plotting ROC curve 

plot(roc_obj, main = "ROC Curve - Logistic Regression")

"The ROC curve visually represents how well this logistic regression model can 
 distinguish between customers who churn and those who stay. 
 The curve plots sensitivity (true positive rate) against 1-specificity (false positive rate) 
 at different classification thresholds. Our curve rises steeply towards the top-left corner,
 showing the model effectively identifies most churn cases while keeping false alarms low.

The area under the ROC curve (AUC) quantifies this performance; values closer to 1 
indicate excellent discrimination. This means model is strong in predicting whether 
a customer will churn, making it a useful tool for targeted retention strategies."

------- # Interpreting the Model ---------


# Model coefficients
summary(new_model$finalModel)

# Odds ratios
exp(coef(new_model$finalModel))


"The logistic regression model reveals several key insights into customer churn.
Longer contracts, such as one-year or two-year agreements, significantly reduce churn
risk by 70% and 91% respectively, emphasizing the value of customer commitment. 
Online security and premium support services also help lower churn odds by 42% and 28%, 
highlighting the importance of added service features. Higher monthly charges modestly 
increase the risk of churn, while paperless billing surprisingly raises churn likelihood
by 38%. Gender differences are observed, with males having an 18% lower churn risk.
Internet type matters too, as DSL users tend to stay longer compared to others.
Overall, these findings guide targeted strategies for improving customer retention through 
contract incentives, service enhancements, and personalized offerings."

--------------------------- MARKET BASKET ANALYSIS -----------------------------

# Installing and Loading Required Packages

install.packages("arules")
library(arules)

install.packages("arulesViz")
library(arulesViz)

update.packages(ask = FALSE, checkBuilt = TRUE)












