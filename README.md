# Food-Classification-Using-Nutritional-Data

In the era of increasing dietary awareness, the ability to classify food items based on nutritional attributes are invaluable. This project involves developing a **machine learning model** that classifies food into multiple categories such as calories, proteins, carbohydrates, fats, and sugar. By the end of this project,  we will be able to create a **robust classification system** that can accurately label food types and gain insights into what makes each food category distinct. management systems.

The initial step in our machine learning pipeline involved acquiring and preparing the dataset. The dataset was downloaded from the given source, extracted, and successfully imported into a Pandas DataFrame for further analysis and manipulation.

**Handling Missing and Inconsistent Data** To ensure data quality, we began by inspecting the dataset for null, empty, or anomalous values. This included checking for NaN's and other placeholders that may indicate missing data. Where applicable, imputation techniques were applied to fill in missing values, using methods appropriate to the data type and distribution (e.g., mean or median imputation).

We also performed value counts on individual features to uncover uncommon, unusual, or inconsistent entries. These checks helped in detecting possible data entry errors or formatting issues that could distort model performance. Any inconsistent or unknown values were cleaned or corrected to maintain data integrity.

**Data Formatting and Structure** Next, we standardized column names and formats to ensure uniformity throughout the dataset. This step involved correcting inconsistencies in naming conventions, aligning data types, and organizing the dataset for seamless analysis and model input. Since the dataset consisted of continuous numerical features and few of them with categorical string formats, based on that numerical features were standardized, and label encoding was performed on the categorical features.

**Feature Selection and Correlation Analysis** To focus on the most impactful variables, we conducted a correlation analysis. This helped identify and retain features that had strong relationships with the target variable, and also PCA was applied as a dimensionality reduction technique to select relevant features for the development of ML models.

**Definition of Target Variable** The target variable, Food Name, is a categorical variable used to classify food items based on their nutritional characteristics.

**Data Splitting** To evaluate the model’s generalizability, the dataset was split into training and testing subsets. This allows the model to learn patterns from the training data and validate its performance on previously unseen data. A standard train-test split ratio (e.g., 80/20 or 70/30) was used to ensure a fair evaluation.

**Model Training and Prediction** Logistic Regression, Random Forest, KNN, Decision Tree, Gradient Booting and X-Gradient Boosting model was trained using the selected input features. After training, the model was used to generate predictions on both the training and test sets. This step helped assess whether the model was underfitting or overfitting the data and also help to find the suitable model for the given data set.

**Performance Evaluation** To quantify the model's performance, we used classification metrics Accuracy, Precision, Recall, F1-Score as the primary evaluation metric. 
**Accuracy:** To Measures overall correctness of the model.
**Precision:** To Measures how many predicted positives are actually correct.
**Recall:** To Measures how many actual positives are correctly identified.
**F1-Score:** To Balances precision and recall.
If all metrics are high, the model is performing well, correctly identifying and predicting the classes with minimal error.
