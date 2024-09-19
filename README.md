# Executive Summary:
This project aims to classify mushrooms as either edible or poisonous based on various physical characteristics. A machine learning model was developed using the LightGBM algorithm, which achieved high accuracy in predicting mushroom edibility.

# Data Overview:
The dataset consists of training and test sets, with various features describing mushroom attributes such as cap shape, cap surface, gill attachment, and stem characteristics. The target variable is 'class', indicating whether a mushroom is **edible ('e')** or **poisonous ('p')**.

# Data Preprocessing:
1. Duplicate records were removed from the training dataset.
2. Features with more than **20% null** values were **identified and removed**.
3. Categorical features were cleaned by **replacing NA values with 'missing'** and **grouping low-frequency categories (count < 100) as 'noise'**.
4. Numerical features, such as **cap diameter, were imputed with mean values**.

# Exploratory Data Analysis:
Several visualizations were created to understand the data distribution and relationships:

1. **Count plots for categorical features:**

   These plots showed the distribution of each category within features, split by the target class. For example, certain cap shapes or gill attachments might be more common in poisonous or edible mushrooms.

![bargraph16](https://github.com/user-attachments/assets/254d5ace-2a2b-41b0-a2f2-54c15e9685b8)
![bargraph15](https://github.com/user-attachments/assets/5b518200-770e-4906-8a4f-0694fcbd7a1d)
![bargraph14](https://github.com/user-attachments/assets/16e086ce-0d9a-47e5-8223-249e15d56714)


3. **Histograms for numerical features:**
   These plots displayed the distribution of numerical features like cap diameter, showing how they differ between edible and poisonous mushrooms.
![hist3](https://github.com/user-attachments/assets/75508f4b-4305-4799-9117-a0232e943838)



5. **Box plots and violin plots for numerical features:**
   These visualizations provided insights into the range and distribution of numerical features for each class, highlighting potential differences between edible and poisonous mushrooms.

![violin3](https://github.com/user-attachments/assets/41411bb0-ff31-42e0-a225-9bbcaaa644a1)

![box3](https://github.com/user-attachments/assets/fe29d06b-07a6-4cb7-b356-9d63f6948184)




7. **Target distribution plot:**
   This showed the overall balance between edible and poisonous mushrooms in the dataset.

![targetdist](https://github.com/user-attachments/assets/c179f19e-4be0-4230-9ce9-6f89fbd207e5)


# Key observations from these visualizations:
- _As the size increases for cap, stem width, and height, there seems to be a higher correlation with poisonous mushrooms._
- _Certain categorical features showed clear distinctions between edible and poisonous mushrooms, which could be strong predictors in the model._

# Model Development:
The LightGBM algorithm was chosen for this classification task. Hyperparameter tuning was performed using Optuna, an hyperparameter optimization framework. The best parameters found were:

```python
lgb_params = {
    'lambda_l1': 4.9821954607856234e-08,
    'lambda_l2': 0.7502003319916978,
    'num_leaves': 256,
    'feature_fraction': 0.46578010078293625,
    'bagging_fraction': 0.9390561952589022,
    'bagging_freq': 1,
    'min_child_samples': 93,
    'device': 'cpu'
}
```

# Model Performance:
The model achieved a high **Matthews Correlation Coefficient (MCC) score of 0.9847 during cross-validation**, indicating excellent performance in distinguishing between edible and poisonous mushrooms.

# Predictions and Submission:
The trained model was used to make predictions on the test set. **The binary predictions (0 or 1) were mapped to 'e' for edible and 'p' for poisonous mushrooms**. A submission file was created with these predictions.

# Conclusions and Recommendations:
1. The **LightGBM model shows strong performance in classifying mushrooms, with a high MCC score**.
2. Feature importance analysis could provide insights into which mushroom characteristics are most crucial for determining edibility.
3. Further validation of **new, unseen data would be beneficial to ensure the model's generalizability**.
4. **Consider ensemble methods or stacking** with other algorithms to improve performance further.
5. Develop a user-friendly interface for practical applications, allowing users to input mushroom characteristics and receive edibility predictions.

# Future Work:
1. Explore more advanced feature engineering techniques.
2. Investigate the possibility of **multi-class classification** if more detailed mushroom categories are available.
3. Collect more data on rare mushroom species to improve model robustness.
4. Develop a **mobile application for real-time mushroom classification in the field**
