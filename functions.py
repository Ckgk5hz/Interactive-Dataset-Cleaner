import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

### Standardization functions ###
def lower(df_cleaned, col):
    return  df_cleaned[col].str.lower()

def upper(df_cleaned, col):
    return df_cleaned[col].str.upper()

def title(df_cleaned, col):
    return df_cleaned[col].str.title()

def date_time(df_cleaned, col):
    return pd.to_datetime(df_cleaned[col], errors='coerce')

def rounding(df_cleaned, col, decimal_places):
    return df_cleaned[col].round(decimal_places)

def drop_missing_values(df_cleaned):
    return df_cleaned.dropna()

def mean(df_cleaned, col):
    return df_cleaned[col].fillna(df_cleaned.mean(numeric_only=True))

def median(df_cleaned, col):
    return df_cleaned[col].fillna(df_cleaned.median(numeric_only=True))

def mode(df_cleaned, col2):
    return df_cleaned[col2].fillna(df_cleaned[col2].mode()[0])

def custom(df_cleaned, col, custom_value):
    return df_cleaned[col].fillna(custom_value)

def drop_duplicates(df_cleaned):
    return df_cleaned.drop_duplicates()

### One Hot Encoding function ###

def One_hot_encoding(df_cleaned, onehot_columns):

    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids multicollinearity
    # Fit and transform the selected columns
    encoded_data = encoder.fit_transform(df_cleaned[onehot_columns])
    
    # Create a DataFrame for the encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(onehot_columns))
    
    # Drop original categorical columns and concatenate the one-hot encoded columns
    df_cleaned = df_cleaned.drop(onehot_columns, axis=1)
    df_cleaned = pd.concat([df_cleaned, encoded_df], axis=1)

    return df_cleaned

def normalize(df_cleaned, col):
    df_cleaned[col] = (df_cleaned[col] - df_cleaned[col].min()) / (df_cleaned[col].max() - df_cleaned[col].min())
    return df_cleaned[col]

def outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1  # Interquartile range

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the dataset by removing outliers
    df_cleaned = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df_cleaned

def fit_linear_regression(df_cleaned, reg_cols):
    """Fit a linear regression model using specified columns."""
    reg = linear_model.LinearRegression()
    var1 = df_cleaned[[reg_cols[0], reg_cols[1]]]  # Predictor variables
    model = reg.fit(var1, df_cleaned[reg_cols[2]])  # Target variable

    return model  # Return the trained model

def lin_regressor(df_cleaned, reg_cols, values):
    model = fit_linear_regression(df_cleaned, reg_cols)
    return model.predict(values)

def fit_poly_regression(df_cleaned, reg_cols, degree):
    poly = PolynomialFeatures(degree)
    var1 = df_cleaned[[reg_cols[0], reg_cols[1]]]
    var1_poly = poly.fit_transform(var1)
    model = linear_model.LinearRegression().fit(var1_poly, df_cleaned[[reg_cols[2]]])
    return model, poly, var1_poly

def poly_regressor(df_cleaned, reg_cols, degree, values):
    model, poly, _ = fit_poly_regression(df_cleaned, reg_cols, degree)
    return model.predict(poly.fit_transform(values))

def fit_knn(df_cleaned, features, target, k):
    X = df_cleaned[features].values
    y = df_cleaned[target].values
    knn = KNeighborsClassifier(n_neighbors=k)
    model = knn.fit(X, y)
    
    return model, X, y  # Return the model and the feature/target array

def knn_(df_cleaned, reg_cols, k, values):
    model, _, m = fit_knn(df_cleaned, reg_cols, k)
    return model.predict(values)