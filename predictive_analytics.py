# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_csv('supermarket_sales.csv')
df.info()

df.describe()
df.duplicated().sum()
df.isnull().sum()

date_columns = ['Date']
for column in date_columns:
  df[column] = pd.to_datetime(df[column])

df.info()

# EDA
# Fungsi untuk mendeteksi outliers menggunakan IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    print(f"📌 {column} - Jumlah Outliers: {len(outliers)}")
    return outliers

# Pilih kolom numerik untuk dicek outliers
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Cek outliers di setiap kolom numerik
outlier_dict = {}
for col in numerical_columns:
    outlier_dict[col] = detect_outliers_iqr(df, col)

# Plot boxplot untuk visualisasi outliers
plt.figure(figsize=(12, 6))
df[numerical_columns].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot untuk Deteksi Outliers")
plt.show()

cols_with_outliers = ["Tax 5%", "Total"]

plt.figure(figsize=(6, 6))
for col in cols_with_outliers:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot {col}")
    plt.show()
  
# Univariate Analysis
# Pisahkan categorical & numerical features
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Univariate Analysis untuk Categorical Features
def analyze_categorical_features(df, categorical_cols):
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, palette="viridis", order=df[col].value_counts().index)
        plt.title(f"Distribusi {col}")
        plt.xticks(rotation=45)
        plt.show()
        print(f"\n📌 {col} - Unique Values:\n", df[col].value_counts())
        print("-" * 50)

# Univariate Analysis untuk Numerical Features
def analyze_numerical_features(df, numerical_cols):
    for col in numerical_cols:
        plt.figure(figsize=(12, 5))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, bins=30, color='blue')
        plt.title(f"Histogram {col}")

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col], color='red')
        plt.title(f"Boxplot {col}")

        plt.show()
        print(f"\n📊 {col} - Statistik Deskriptif:\n", df[col].describe())
        print("-" * 50)

# Panggil fungsi analisis
print("\n=== 🎯 Univariate Analysis untuk Categorical Features ===")
analyze_categorical_features(df, categorical_cols)

print("\n=== 📈 Univariate Analysis untuk Numerical Features ===")
analyze_numerical_features(df, numerical_cols)

# Multivariate Analysis
# Pisahkan categorical & numerical features
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Multivariate Analysis untuk Categorical Features
def analyze_categorical_multivariate(df, categorical_cols, target_col):
    for col in categorical_cols:
        if col != target_col:  # Hindari memplot target sebagai independent
            plt.figure(figsize=(8, 4))
            sns.boxplot(data=df, x=col, y=target_col, palette="viridis")
            plt.title(f"{target_col} vs {col}")
            plt.xticks(rotation=45)
            plt.show()

            print(f"\n📌 {col} vs {target_col} - Statistik Grup:\n", df.groupby(col)[target_col].describe())
            print("-" * 50)

# Multivariate Analysis untuk Numerical Features
def analyze_numerical_multivariate(df, numerical_cols):
    # Pairplot untuk melihat hubungan antar numerical features
    plt.figure(figsize=(12, 6))
    sns.pairplot(df[numerical_cols], diag_kind="kde")
    plt.show()

    # Heatmap Korelasi
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap Korelasi")
    plt.show()

# Tentukan target untuk analisis kategori 
target_col = "Total"  

print("\n=== 🎯 Multivariate Analysis untuk Categorical Features vs Target ===")
analyze_categorical_multivariate(df, categorical_cols, target_col)

print("\n=== 📈 Multivariate Analysis untuk Numerical Features ===")
analyze_numerical_multivariate(df, numerical_cols)

# Data Preparation
# Encoding Categorical features
from sklearn.preprocessing import OneHotEncoder

df = pd.concat([df, pd.get_dummies(df['Branch'], prefix='Branch')], axis=1)
df = pd.concat([df, pd.get_dummies(df['City'], prefix='City')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Customer type'], prefix='Customer type')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Gender'], prefix='Gender')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Product line'], prefix='Product line')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Payment'], prefix='Payment')], axis=1)
df.drop(['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment'], axis=1, inplace=True)
df.head()

# Reduksi Dimensi using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=123)
pca.fit(df[['Unit price', 'Tax 5%']])
princ_comp = pca.transform(df[['Unit price', 'Tax 5%']])

pca.explained_variance_ratio_.round(2)

pca = PCA(n_components=1, random_state=123)
pca.fit(df[['Unit price','Tax 5%']])
df['dimension'] = pca.transform(df.loc[:, ('Unit price','Tax 5%')]).flatten()
df.drop(['Unit price','Tax 5%'], axis=1, inplace=True)
df.head()

# Pembagian Data Train dan Test
X = df.drop(["Total"],axis =1)
y = df["Total"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

print(f'Total of sample in whole dataset: {len(X)}')
print(f'Total of sample in train dataset: {len(X_train)}')
print(f'Total of sample in test dataset: {len(X_test)}')

# Standarisasi
numerical_features = ['Quantity', 'Rating', 'dimension']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(4)

# Drop Kolom Date 
X_train = X_train.drop(columns=['Date'])
X_test = X_test.drop(columns=['Date'])

# Model Development
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])

# K-NN model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
 
models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

# Random Forest model
from sklearn.ensemble import RandomForestRegressor
 
RF = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=123, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

# AdaBoost Model
from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# Model Evaluation
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e2
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e2

mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
