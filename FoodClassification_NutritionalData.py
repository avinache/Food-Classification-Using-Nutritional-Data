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














