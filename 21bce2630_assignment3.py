#21BCE2630_Shrey_Gupta_Evening_Batch_AIML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import f_oneway

#TASK 1 - Downloading the dataset (Done)
#TASK 2 - Loading the dataset
df = pd.read_csv('"C:\Users\Shrey Gupta\Desktop\Academics\AIML\penguins_size.csv"')
df.head()

#TASK 3 - Visualizations
#Univariate analysis (culmen length)
sns.histplot(df.culmen_length_mm)
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Frequency')
plt.title('Histogram of Culmen Length')

#Bivariate analysis (culmen length vs depth)
sns.scatterplot(x = 'culmen_length_mm', y = 'culmen_depth_mm', data = df)
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Culmen Depth (mm)')
plt.title('Scatter Plot between Culmen Length and Culmen Depth')

#Multivariate analysis
sns.pairplot(df)

#TASK 4 - Descriptive statistics
df.describe()

#TASK 5 - Checking for missing values
df.isnull().sum()

#null values in columns:-
#culmen_length_mm (2 null values)
#culmen_depth_mm (2 null values)
#flipper_length_mm (2 null values)
#body_mass_g (2 null values)
#sex (10 null values)

#We will replace the null values with the median(for integer values) or mode(for non-integer values) of their respective column
#Calculating the median/mode for the afore-mentioned columns
median_value_1 = df['culmen_length_mm'].median()
median_value_2 = df['culmen_depth_mm'].median()
median_value_3 = df['flipper_length_mm'].median()
median_value_4 = df['body_mass_g'].median()
mode_value = df['sex'].mode()[0]

#Replacing the null values
df['culmen_length_mm'].fillna(median_value_1, inplace = True)
df['culmen_depth_mm'].fillna(median_value_2, inplace = True)
df['flipper_length_mm'].fillna(median_value_3, inplace = True)
df['body_mass_g'].fillna(median_value_4, inplace = True)
df['sex'].fillna(mode_value, inplace = True)

#Checking now for null values
df.isnull().sum()

#TASK 6 - Finding and replacing outliers
mean_body_mass = df['body_mass_g'].mean()
std_body_mass = df['body_mass_g'].std()
df['body_mass_g'] = np.where((df['body_mass_g'] > mean_body_mass + 3 * std_body_mass) | (df['body_mass_g'] < mean_body_mass - 3 * std_body_mass), mean_body_mass, df['body_mass_g'])

#Data after outlier treatment
plt.subplot(1, 2, 2)
sns.histplot(df['body_mass_g'], bins=20, kde=True)
plt.xlabel('Data after Outlier Treatment')
plt.ylabel('Frequency')
plt.title('Histogram of Data after Outlier Treatment')

#The graph is the same as before, hence, there are no outliers.

#TASK 7 - Checking correlation of independent variable with target
from scipy.stats import f_oneway
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['species'] = le.fit_transform(df['species'])
df['island'] = le.fit_transform(df['island'])
df.head()

# Task 8 - Check for categorical columns and perform encoding
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

for column, encoder in label_encoders.items():
    print(f"Label Encoding for {column}:")
    for class_label, encoded_label in enumerate(encoder.classes_):
        print(f"{class_label}: {encoded_label}")

#TASK 9 - Splitting the data into dependent and independent variables
X = df.drop(columns=['species'])  # Independent variables
y = df['species']  # Dependent variable
print("Independent Variables (X) Shape:", X.shape)
print("Dependent Variable (y) Shape:", y.shape)

#TASK 10 - Scaling the data
X = df.drop('species', axis=1)
y = df['species']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled Data:")
print(X_scaled)

#TASK 11 - Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#TASK 12 - Checking the shapes of training and testing data
print("Training Data Shape (X_train):", X_train.shape)
print("Training Data Shape (y_train):", y_train.shape)
print("Testing Data Shape (X_test):", X_test.shape)
print("Testing Data Shape (y_test):", y_test.shape)