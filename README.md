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

![bargraph17](https://github.com/user-attachments/assets/3539381b-6a50-4854-98f6-13c1e030ef3b)
![bargraph16](https://github.com/user-attachments/assets/8bbf1228-3b9d-47c6-9375-e9cfb9245804)
![bargraph15](https://github.com/user-attachments/assets/8dcc23cb-d73b-48be-9ce2-c5497cecc2d9)
![bargraph14](https://github.com/user-attachments/assets/07b1cad0-027c-40c4-8355-40cdd49ef963)
![bargraph13](https://github.com/user-attachments/assets/66806d8d-6882-40ac-9337-3fc8e8be24a5)
![bargraph12](https://github.com/user-attachments/assets/00ad9d11-2ba2-4cd9-b79d-ca1a3f2766d7)
![bargraph11](https://github.com/user-attachments/assets/ab9ff79f-1776-4b8e-9f17-81a742160345)
![bargraph10](https://github.com/user-attachments/assets/c6c65b62-cf6b-4f08-a0c5-972285b4c850)
![bargraph9](https://github.com/user-attachments/assets/a3bc00a4-9e20-4cf1-907c-b494cc94dcda)
![bargraph8](https://github.com/user-attachments/assets/c552df10-334d-4212-afe5-d8ad7352f892)
![bargraph7](https://github.com/user-attachments/assets/7ca90d36-d563-4bf3-bf76-5d86bcdca0c6)
![bargraph5](https://github.com/user-attachments/assets/47951f8c-43f6-4a30-8329-a7655d43cdd4)
![bargraph4](https://github.com/user-attachments/assets/a854fb45-03a4-4dbd-9eb3-c949b5e1447f)
![bargraph3](https://github.com/user-attachments/assets/5ad2baea-40c3-4b18-8bf9-65f61fa45173)
![bargraph2](https://github.com/user-attachments/assets/b626cfb9-3319-43e2-9e7d-80e46f28dc68)
![bargraph1](https://github.com/user-attachments/assets/4139546e-3489-43a1-a372-6a5220aff980)


3. **Histograms for numerical features:**
   These plots displayed the distribution of numerical features like cap diameter, showing how they differ between edible and poisonous mushrooms.
   ![hist3](https://github.com/user-attachments/assets/42ba8e74-ff9f-42d4-adb1-d101b73bd226)
![hist2](https://github.com/user-attachments/assets/2beb8593-e1a3-4c9f-a258-b98940ec79af)
![hist1](https://github.com/user-attachments/assets/525d8d2b-9267-417b-a99e-c3a5f8bc5550)


5. **Box plots and violin plots for numerical features:**
   These visualizations provided insights into the range and distribution of numerical features for each class, highlighting potential differences between edible and poisonous mushrooms.
   ![box3](https://github.com/user-attachments/assets/9a78f2a1-810a-41df-91a4-cbf38503149a)
![box2](https://github.com/user-attachments/assets/387d4bf0-c75e-47b1-926c-ef2828b10d7f)
![box1](https://github.com/user-attachments/assets/673ee7a5-2124-428c-a385-f899e5a44fe9)

![violin3](https://github.com/user-attachments/assets/a8087ae8-04da-4086-b5d6-9513ab351451)
![violin2](https://github.com/user-attachments/assets/7d870004-9da6-43ef-970a-aa044b3932b6)
![violin1](https://github.com/user-attachments/assets/315a6e59-4442-4159-b21c-06424bb9ca3b)



7. **Target distribution plot:**
   This showed the overall balance between edible and poisonous mushrooms in the dataset.
   ![targetdist](https://github.com/user-attachments/assets/2865d3bb-c2c2-45da-a520-e23641fca2ac)


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
