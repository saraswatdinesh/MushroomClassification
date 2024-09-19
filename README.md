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

![bargraph17](https://github.com/user-attachments/assets/6384f23d-c72b-4080-9a04-ddbf95d92161)
![bargraph16](https://github.com/user-attachments/assets/254d5ace-2a2b-41b0-a2f2-54c15e9685b8)
![bargraph15](https://github.com/user-attachments/assets/5b518200-770e-4906-8a4f-0694fcbd7a1d)
![bargraph14](https://github.com/user-attachments/assets/16e086ce-0d9a-47e5-8223-249e15d56714)
![bargraph13](https://github.com/user-attachments/assets/8d476799-42b0-4056-b592-fc7951f79681)
![bargraph12](https://github.com/user-attachments/assets/722cd4f9-c756-49fc-81bf-f3919b2f0291)
![bargraph11](https://github.com/user-attachments/assets/471cfb13-b65f-4616-bef9-c121f0971bf9)
![bargraph10](https://github.com/user-attachments/assets/d0ec5c03-3125-4761-a196-6b3454197f82)
![bargraph9](https://github.com/user-attachments/assets/3eb68c6c-3f16-4681-839e-ccfd907b1aa4)
![bargraph8](https://github.com/user-attachments/assets/65ffa32a-b30b-4544-a8da-8557b28a94b6)
![bargraph7](https://github.com/user-attachments/assets/e3298410-caea-4936-ad5f-b2019fc40acd)
![bargraph5](https://github.com/user-attachments/assets/d7e56bff-cf0c-4148-845c-96c46de048cb)
![bargraph4](https://github.com/user-attachments/assets/b8129ca4-c0e3-44c6-9b08-0817c23a0f98)
![bargraph3](https://github.com/user-attachments/assets/599faf9f-52db-4fd0-9e3a-abe2bb5a4f5e)
![bargraph2](https://github.com/user-attachments/assets/a1153ed1-10d0-4f61-b00b-80cce8f9849d)
![bargraph1](https://github.com/user-attachments/assets/4e34e966-c151-4acc-9d0d-a1bd3460fca1)


3. **Histograms for numerical features:**
   These plots displayed the distribution of numerical features like cap diameter, showing how they differ between edible and poisonous mushrooms.
![hist3](https://github.com/user-attachments/assets/75508f4b-4305-4799-9117-a0232e943838)
![hist2](https://github.com/user-attachments/assets/15b92c6c-ad06-4dde-ba01-f3db171da263)
![hist1](https://github.com/user-attachments/assets/e3832044-ace7-448a-afd5-6184d03f45e8)



5. **Box plots and violin plots for numerical features:**
   These visualizations provided insights into the range and distribution of numerical features for each class, highlighting potential differences between edible and poisonous mushrooms.

![violin3](https://github.com/user-attachments/assets/41411bb0-ff31-42e0-a225-9bbcaaa644a1)
![violin2](https://github.com/user-attachments/assets/d1908e14-7dc9-48cf-8e20-5686fadb1826)
![violin1](https://github.com/user-attachments/assets/eef9ff56-bbed-447d-8472-8d23d7ea7bd3)
![box3](https://github.com/user-attachments/assets/fe29d06b-07a6-4cb7-b356-9d63f6948184)
![box2](https://github.com/user-attachments/assets/b4c22230-6ced-418e-8258-84e11393b9d3)
![box1](https://github.com/user-attachments/assets/5ff0f5ac-6526-4b1c-8bd8-123410435b38)



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
