import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
import warnings
warnings.filterwarnings('ignore')

# LOADING DATASETS
product_df = pd.read_csv('content/product_info.csv')
review_df_01 = pd.read_csv('content/reviews_0-250.csv', index_col = 0, dtype={'author_id':'str'})
review_df_02 = pd.read_csv('content/reviews_250-500.csv', index_col = 0, dtype={'author_id':'str'})
review_df_03 = pd.read_csv('content/reviews_500-750.csv', index_col = 0, dtype={'author_id':'str'})
review_df_04 = pd.read_csv('content/reviews_750-1250.csv', index_col = 0, dtype={'author_id':'str'})
review_df_05 = pd.read_csv('content/reviews_1250-end.csv', index_col = 0, dtype={'author_id':'str'})

# MERGIG ALL REVIEWS DATAFRAMES
review_df = pd.concat([review_df_01, review_df_02, review_df_03, review_df_04, review_df_05], axis=0)

# CHECKING COLUMNS THAT ARE COMMON IN BOTH DATAFRAMES
cols_to_use = product_df.columns.difference(review_df.columns)
cols_to_use = list(cols_to_use)
cols_to_use.append('product_id')
print(cols_to_use)

# AS DATAFRAMES HAVE COMMON COLUMN 'product_id', WE CAN MERGE THEM ON 'product_id'
df = pd.merge(review_df, product_df[cols_to_use], how='outer', on=['product_id', 'product_id'])
df = df.iloc[:100000]
cols = """variation_desc
sale_price_usd
value_price_usd
child_max_price
child_min_price
review_title"""
cols_list = cols.split("\n")
df.drop(columns=cols_list,axis=1,inplace=True)

# DROP ROWS WITH MISSING VALUES
df.dropna(axis=0,inplace=True)
df.drop(columns=['submission_time'], axis=1, inplace=True)

# ONE-HOT ENCODING CATEGORICAL VARIABLES
categorical_columns = ['skin_tone','eye_color', 'hair_color', 'primary_category', 'secondary_category', 'size', 'tertiary_category', 'variation_type', 'variation_value', 'skin_type']
df = pd.get_dummies(df, columns=categorical_columns)

# Scaling numerical features
scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
# Feature Selection
X = df.drop(columns=['author_id', 'review_text', 'product_id', 'rating', 'highlights', 'ingredients',
                     'product_name', 'brand_name'])
y = df['rating']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define a function to create the model with dropout_rate parameter
def create_model(dropout_rate=0.5, optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation=activation))
    model.add(Dropout(dropout_rate))  # Dropout rate as parameter
    model.add(Dense(64, activation=activation))
    model.add(Dropout(dropout_rate))  # Dropout rate as parameter
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model


# Wrap the model with KerasRegressor for scikit-learn
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define hyperparameters to tune
param_grid = {
    'batch_size': [32, 64, 128],
    'epochs': [50, 100, 200],
    'model__optimizer': ['adam', 'rmsprop'],
    'model__activation':['relu','softmax'],
    'model__dropout_rate': [0.3, 0.5, 0.7]
}

# Randomized Search for hyperparameters
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=3, verbose=1,
                                   random_state=42)
random_search_result = random_search.fit(X_train, y_train)

# Best Model
best_model = random_search_result.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
accuracy = np.mean(np.abs(y_pred - y_test) <= 0.5)

print(f'Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Accuracy within ±0.5: {accuracy:.4f}')
