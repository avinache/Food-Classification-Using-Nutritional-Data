import pandas as pd
import numpy as np
import os
import json
import mysql.connector as db
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder

mm = MinMaxScaler()
ss = StandardScaler()

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Read the file into a DataFrame
data = pd.read_csv("synthetic_food_dataset_imbalanced.csv")

#Drop NaN values from data set
data.dropna(inplace=True)

#Copy data set
AnalysisData = data

# Average calories based on meal type
categorical_cols = ['Meal_Type', 'Preparation_Method', 'Is_Vegan', 'Is_Gluten_Free', 'Food_Name']
AnalysisData[categorical_cols] = AnalysisData[categorical_cols].astype('category')
AnalysisData.groupby('Meal_Type')['Calories'].mean().reset_index()


# Average Calories by Meal Type
fig = px.bar(
    AnalysisData.groupby('Meal_Type', as_index=False)['Calories'].mean(),
    x='Meal_Type',
    y='Calories',
    color='Meal_Type',
    title='Average Calories by Meal Type',
)
fig.update_layout(
    yaxis_title='Average Calories',
    xaxis_title='Meal Type',
    title_x=0.5
)
fig.show()


#  Calories vs Protein for Food Categories
fig = px.scatter(
    AnalysisData,
    x='Protein',
    y='Calories',
    color='Food_Name',
    symbol='Meal_Type',
    size_max=100,
    title='Calories vs. Protein by Food Item and Meal Type',
    labels={'Protein': 'Protein (g)', 'Calories': 'Calories'}
)
fig.show()

# Nutrient Distribution for Vegan vs Non-Vegan
nutrients = ['Calories', 'Protein', 'Fat', 'Carbs', 'Fiber']
vegan = AnalysisData[AnalysisData['Is_Vegan'] == True]
Non_vegan = AnalysisData[AnalysisData['Is_Vegan'] == False]

avg_vegan = vegan[nutrients].mean()
avg_non_vegan = Non_vegan[nutrients].mean()

NutrientData = pd.DataFrame({'Vegan': avg_vegan, 'Non-Vegan': avg_non_vegan})
# Reshape the DataFrame for Plotly (from wide to long format)
NutrientData_long = NutrientData.reset_index().melt(
    id_vars='index',
    var_name='Food_Type',
    value_name='Average_Value'
).rename(columns={'index': 'Nutrient'})

# Create the bar plot
fig = px.bar(
    NutrientData_long,
    x='Nutrient',
    y='Average_Value',
    color='Food_Type',
    barmode='group',
    title='Nutrient Comparison: Vegan vs Non-Vegan Foods',
    labels={'Average_Value': 'Average Value'}
)
fig.update_layout(
    xaxis_title='Nutrient',
    yaxis_title='Average Value',
)
fig.show()

# Glycemic Index vs Sugar Content by Meal Type
fig = px.scatter(
    AnalysisData,
    x='Sugar',
    y='Glycemic_Index',
    color='Meal_Type',
    size='Calories',
    size_max=50,
    title='Glycemic Index vs Sugar Content by Meal Type',
    labels={
        'Sugar': 'Sugar (g)',
        'Glycemic_Index': 'Glycemic Index'
    }
)
fig.update_layout(
    xaxis_title='Sugar (g)',
    yaxis_title='Glycemic Index'
)
fig.show()

# Correlation Heatmap of Nutritional Features 
nutritional_cols = ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar', 'Fiber', 
                    'Sodium', 'Cholesterol', 'Glycemic_Index', 'Water_Content']
plt.figure(figsize=(15, 5))
sns.heatmap(AnalysisData[nutritional_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Nutritional Features')
plt.show()

# ****** Balanced Meal Recommendation ******
recommended = AnalysisData[
    (AnalysisData['Calories'] >= 200) & (AnalysisData['Calories'] <= 350) &
    (AnalysisData['Protein'] >= 10) &
    (AnalysisData['Fiber'] >= 2) &
    (AnalysisData['Sugar'] <= 6)
]

# Assuming AnalysisData is a DataFrame already loaded
macros = ['Protein', 'Fat', 'Carbs']
df_macros = AnalysisData[macros]

# Melt the dataframe to long format for Plotly Express
df_long = df_macros.melt(var_name='Macronutrient', value_name='Amount (grams)')

# Create box plot
fig = px.box(df_long, x='Macronutrient', y='Amount (grams)', title='Macronutrient Distribution')
fig.show()


# Education: Healthier vs. Less Healthy Criteria
# Define healthier as low sugar, high fiber, lower calories
AnalysisData['Health_Score'] = (1/AnalysisData['Calories']) + AnalysisData['Fiber'] - (AnalysisData['Sugar']/10)
AnalysisDatasorted = AnalysisData.sort_values("Health_Score", ascending=False)
AnalysisDatasorted[['Food_Name', 'Calories', 'Fiber', 'Sugar', 'Health_Score']].head()



#cap outliers using IQR method
#Instead of removing outliers, limit them to maximum or minimum threshold

def cap_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)   # Q1 and Q3 are the 25th and 75th percentiles
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1                 # IQR is the Interquartile Range 
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR  # lower_bound and upper_bound are thresholds used to detect outliers
        df[col] = np.where(df[col] < lower_bound, lower_bound,
                  np.where(df[col] > upper_bound, upper_bound, df[col]))   # Cap the Outliers
    return df

#If a value in the column is below the lower bound, it's set to lower_bound.
#If a value is above the upper bound, it's set to upper_bound.
#If it's within bounds, it stays unchanged.
#This process is often called winsorizing — it reduces the impact of outliers without deleting data.

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
caped_Data = cap_outliers(data, numeric_cols)

cr = caped_Data[numeric_cols].corr()
plt.figure(figsize=(14, 5))
sns.heatmap(cr, cmap= 'coolwarm', annot= True)
plt.show()

#Normalize Numerical Features
scaler = StandardScaler()
caped_Data[numeric_cols] = scaler.fit_transform(caped_Data[numeric_cols])

cr = caped_Data[numeric_cols].corr()
plt.figure(figsize=(14, 5))
sns.heatmap(cr, cmap= 'coolwarm', annot= True)
plt.show()

# Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)
pca_features = pca.fit_transform(caped_Data[numeric_cols])

df_pca_encoded = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(pca_features.shape[1])])

#Copy dataset
RData=caped_Data

# Label Encoding
labels = caped_Data[['Meal_Type','Preparation_Method','Is_Vegan','Is_Gluten_Free','Food_Name']]
labels.head()
labels = labels.astype(str)
encoded_labels = labels.copy()
labels = labels.drop('Food_Name', axis=1)
label_encoder = LabelEncoder()
for col in labels.columns:
    encoded_labels[col] = label_encoder.fit_transform(labels[col])

combined_col = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'Meal_Type', 'Preparation_Method', 'Is_Vegan', 'Is_Gluten_Free', 'Food_Name']
df_pca_encoded_list = df_pca_encoded.values.tolist()
encoded_labels_list = encoded_labels.values.tolist()
# Combine both datasets row-wise
combined_data = [pca + label for pca, label in zip(df_pca_encoded_list, encoded_labels_list)]
combined_df = pd.DataFrame(combined_data, columns=combined_col)

# ********* LOGISTIC ALGORITHM *********
#Set value and target for the ML model
val  = combined_df.drop('Food_Name', axis = 1)
tar = combined_df['Food_Name']

# train and test split
traindata, testdata, trainlab, testlab = train_test_split( val, tar, test_size= 0.20, random_state= 66)
# Train model
model = LogisticRegression()
model.fit(traindata, trainlab)
# Predict
tr_pred = model.predict(traindata)
ts_pred = model.predict(testdata)


# Define function for classification metrics
# Use one of these depending on what you need
# average='macro' # treats all classes equally
# average='micro' # global accuracy over all predictions
# average='weighted'# accounts for class imbalance
def perfromance(lab, pred):
    a = accuracy_score(lab, pred)
    p = precision_score(lab, pred, average='macro')
    r = recall_score(lab, pred, average='macro')
    f = f1_score(lab, pred, average='macro')
    return pd.DataFrame({'Acc': [a], "Precision": [p], "recall": [r], "F1 Score": [f]})

# Training Perfromance
perfromance(trainlab, tr_pred)
# Testing Perfromance
perfromance(testlab, ts_pred)


# ********* DECISION TREE ALGORITHM *********
#Set value and target for the ML model
val  = combined_df.drop('Food_Name', axis = 1)
tar = combined_df['Food_Name']
# train and test split
traindata, testdata, trainlab, testlab = train_test_split( val, tar, test_size= 0.20, random_state= 66)
# Model Training
dt_model = DecisionTreeClassifier(random_state= 56)
dt_model.fit(traindata, trainlab)
# Predict
tr_pred = dt_model.predict(traindata)
ts_pred = dt_model.predict(testdata)
# Training Perfromance
perfromance(trainlab, tr_pred)
# Testing Perfromance
perfromance(testlab, ts_pred)

# ********* RANDOM FOREST ALGORITHM *********
#Set value and target for the ML model
val  = combined_df.drop('Food_Name', axis = 1)
tar = combined_df['Food_Name']
# train and test split
traindata, testdata, trainlab, testlab = train_test_split( val, tar, test_size= 0.20, random_state= 66)
# Model Training
rf_model = RandomForestClassifier(n_estimators= 500, random_state= 66)
rf_model.fit(traindata, trainlab)
# Predict
tr_pred = rf_model.predict(traindata)
ts_pred = rf_model.predict(testdata)
# Training Perfromance
perfromance(trainlab, tr_pred)
# Testing Perfromance
perfromance(testlab, ts_pred)

# ********* K-NEAREST NEIGHBORS ALGORITHM *********
#Set value and target for the ML model
val  = combined_df.drop('Food_Name', axis = 1)
tar = combined_df['Food_Name']
# train and test split
traindata, testdata, trainlab, testlab = train_test_split( val, tar, test_size= 0.20, random_state= 66)
# Build model
Knn_model = KNeighborsClassifier(n_neighbors= 5)
Knn_model.fit(traindata, trainlab)
# Predict
tr_pred = Knn_model.predict(traindata)
ts_pred = Knn_model.predict(testdata)
# Training Perfromance
perfromance(trainlab, tr_pred)
# Testing Perfromance
perfromance(testlab, ts_pred)

# ********* GRADIENT BOOSTING  ALGORITHM *********
#Set value and target for the ML model
val  = combined_df.drop('Food_Name', axis = 1)
tar = combined_df['Food_Name']
# train and test split
traindata, testdata, trainlab, testlab = train_test_split( val, tar, test_size= 0.20, random_state= 66)
# Built Model
gb_model = GradientBoostingClassifier(n_estimators= 600, random_state= 66, max_depth= 3, learning_rate= 0.01)
gb_model.fit(traindata, trainlab)
# Predict
tr_pred = gb_model.predict(traindata)
ts_pred = gb_model.predict(testdata)
# Training perfromance 
print(classification_report(trainlab, tr_pred))
# Testing Performance 
print(classification_report(testlab, ts_pred))

# ********* X-GRADIENT BOOSTING  ALGORITHM *********
#Set value and target for the ML model
val  = combined_df.drop('Food_Name', axis = 1)
tar = combined_df['Food_Name']
# train and test split
traindata, testdata, trainlab, testlab = train_test_split( val, tar, test_size= 0.20, random_state= 66)
le = LabelEncoder()
trainlab_encoded = le.fit_transform(trainlab)
# Built Model
xgb = XGBClassifier(n_estimators= 600, random_state= 66, max_depth= 3, learning_rate= 0.08)
xgb.fit(traindata, trainlab_encoded)
# Predict
tr_pred = xgb.predict(traindata)
ts_pred = xgb.predict(testdata)
# Training perfromance 
print(classification_report(trainlab, tr_pred))
# Testing Performance 
print(classification_report(testlab, ts_pred))



FCol = ['Meal_Type','Preparation_Method','Is_Vegan','Is_Gluten_Free']
label_encoder = LabelEncoder()
for col in FCol:
    RData[col] = label_encoder.fit_transform(RData[col])

# ********* LOGISTIC ALGORITHM *********
#Set value and target for the ML model
val  = RData.drop('Food_Name', axis = 1)
tar = RData['Food_Name']
# train and test split
traindata, testdata, trainlab, testlab = train_test_split( val, tar, test_size= 0.20, random_state= 66)
# Train model
model = LogisticRegression()
model.fit(traindata, trainlab)
# Predict
tr_pred = model.predict(traindata)
ts_pred = model.predict(testdata)
# Training Perfromance
perfromance(trainlab, tr_pred)
# Testing Perfromance
perfromance(testlab, ts_pred)

# ********* DECISION TREE ALGORITHM *********
# Model Training
dt_model = DecisionTreeClassifier(random_state= 56)
dt_model.fit(traindata, trainlab)
# Predict
tr_pred = dt_model.predict(traindata)
ts_pred = dt_model.predict(testdata)
# Training Perfromance
perfromance(trainlab, tr_pred)
# Testing Perfromance
perfromance(testlab, ts_pred)

# ********* RANDOM FOREST ALGORITHM *********
# Model Training 
rf_model = RandomForestClassifier(n_estimators= 500, random_state= 66)
rf_model.fit(traindata, trainlab)
# Predict
tr_pred = rf_model.predict(traindata)
ts_pred = rf_model.predict(testdata)
# Training Perfromance
perfromance(trainlab, tr_pred)
# Testing Perfromance
perfromance(testlab, ts_pred)


# ********* K-NEAREST NEIGHBORS ALGORITHM *********
# Build model
Knn_model = KNeighborsClassifier(n_neighbors= 5)
Knn_model.fit(traindata, trainlab)
# Predict
tr_pred = Knn_model.predict(traindata)
ts_pred = Knn_model.predict(testdata)
# Training Perfromance
perfromance(trainlab, tr_pred)
# Testing Perfromance
perfromance(testlab, ts_pred)

# ********* GRADIENT BOOSTING  ALGORITHM *********
# Built Model
gb_model = GradientBoostingClassifier(n_estimators= 600, random_state= 66, max_depth= 3, learning_rate= 0.01)
gb_model.fit(traindata, trainlab)
# Predict
tr_pred = gb_model.predict(traindata)
ts_pred = gb_model.predict(testdata)
# Training perfromance 
print(classification_report(trainlab, tr_pred))
# Testing Performance 
print(classification_report(testlab, ts_pred))

# ********* X-GRADIENT BOOSTING  ALGORITHM *********
# Built Model
xgb = XGBClassifier(n_estimators= 600, random_state= 66, max_depth= 3, learning_rate= 0.08)
xgb.fit(traindata, trainlab_encoded)
# Predict
tr_pred = xgb.predict(traindata)
ts_pred = xgb.predict(testdata)
# Training perfromance 
print(classification_report(trainlab, tr_pred))
# Testing Performance 
print(classification_report(testlab, ts_pred))


