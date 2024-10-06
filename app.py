import streamlit as st
import pandas as pd
from nlp_query import * # Importing functions

# Page title
st.title('Interactive Dataset Cleaner')

# File uploader to load datasets
uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel/JSON)", type=["csv", "xlsx", "json"])

if 'cleaned_df' not in st.session_state or st.session_state['cleaned_df'].empty:
    st.write("DataFrame is either missing or empty.")
else:
    # Proceed with your logic when 'cleaned_df' exists and is not empty
    df_cleaned = st.session_state['cleaned_df']
    st.write("DataFrame loaded successfully.")

def update_cleaned_df(df):
    """Update the cleaned DataFrame in session state."""
    st.session_state.cleaned_df = df

if uploaded_file:
    # Detect file type and load dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    
    # Show the raw dataset
    st.subheader('Raw Dataset')
    st.write(df)
    update_cleaned_df(df)

    # Cache the cleaned dataset
    org_df = df.copy()
    df_cleaned = df.copy()

    ### Query related section ###

    query = st.text_input("Enter your natural language query:")
    if query:
        # Generate a structured query using NLP
        intent, column, values = classify_intent_with_columns_and_values(query, df_cleaned.columns)

        result = execute_intent(df_cleaned, intent, column, values)
        
        # Show the result to the user
        st.subheader('Query Result')
        st.write(result)
        # update_cleaned_df(result)


    ### Standardization Options ###

    # Select multiple columns for standardization
    standardize_columns = st.sidebar.multiselect(
        'Select columns to standardize (e.g., text, dates, numbers):', 
        df.columns
    )

    if standardize_columns:
        # Iterate over the selected columns
        for col in standardize_columns:
            # Add a section in the sidebar for each column's standardization options
            st.sidebar.subheader(f'Standardization Options for {col}')
            
            # Option 1: Standardize text (lowercase)
            if st.sidebar.checkbox(f'Standardize {col} to Lowercase'):
                if df_cleaned[col].dtype == 'object':
                    df_cleaned[col] = lower(df,col)
                    st.success(f'Column {col} standardized to lowercase.')
                else:
                    st.warning(f'Column {col} is not a text column and cannot be standardized to lowercase.')

            # Option 2: Standardize text (uppercase)
            if st.sidebar.checkbox(f'Standardize {col} to Uppercase'):
                if df_cleaned[col].dtype == 'object':
                    df_cleaned[col] = upper(df_cleaned, col)
                    st.success(f'Column {col} standardized to uppercase.')
                else:
                    st.warning(f'Column {col} is not a text column and cannot be standardized to uppercase.')

            if st.sidebar.checkbox(f'Standardize {col} to Title Case'):
                if df_cleaned[col].dtype == 'object':
                    df_cleaned[col] = title(df_cleaned, col)
                    st.success(f'Column {col} standardized to title case.')
                else:
                    st.warning(f'Column {col} is not a text column and cannot be standardized to title case.')

            # Option 3: Standardize dates (convert to uniform format)
            if st.sidebar.checkbox(f'Standardize {col} Date Format'):
                try:
                    df_cleaned[col] = date_time(df_cleaned, col)
                    st.success(f'Column {col} standardized to date format.')
                except Exception as e:
                    st.warning(f"Column {col} could not be converted to date format: {e}")

            # Option 4: Standardize numeric values (rounding)
            if st.sidebar.checkbox(f'Round Numeric Values for {col}'):
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    decimal_places = st.sidebar.slider(f'Select decimal places for rounding {col}:', 0, 10, 2)
                    df_cleaned[col] = rounding(df_cleaned, col, decimal_places)
                    st.success(f'Column {col} rounded to {decimal_places} decimal places.')
                else:
                    st.warning(f'Column {col} is not numeric and cannot be rounded.')

        st.write(df_cleaned)
        update_cleaned_df(df_cleaned)

    ### One-Hot Encoding ###
    st.sidebar.subheader(f'Other Options for {uploaded_file.name}')

    # Option for one-hot encoding
    
    if st.sidebar.checkbox('Apply One-Hot Encoding'):
        onehot_columns = st.sidebar.multiselect(
            'Select columns for One-Hot Encoding:', 
            df_cleaned.select_dtypes(include='object').columns
        )
        
        if onehot_columns:
            # Initialize OneHotEncoder
            try:
                df_cleaned = One_hot_encoding(df_cleaned, onehot_columns)
                st.subheader('Dataset after One-Hot Encoding')
                st.write(df_cleaned)
                update_cleaned_df(df_cleaned)
            except Exception as e:
                st.error(f"Error in One-Hot Encoding: {str(e)}")
        else:
            st.warning("No columns selected for One-Hot Encoding.")


    ### Normalization ###

    # Option for normalization
    if st.sidebar.checkbox('Apply Normalization'):
        sidebar_columns = st.sidebar.multiselect(
            'Select columns for Normalization:', 
            df_cleaned.select_dtypes(include='number').columns
        )

        # Check if any columns were selected
        if not sidebar_columns:
            st.warning('No columns selected for Normalization.')
        else:
            try:
                for col in sidebar_columns:
                    df_cleaned[col] = normalize(df_cleaned, col)
                
                st.subheader('Dataset after Normalization')
                st.write(df_cleaned)
                update_cleaned_df(df_cleaned)
            except Exception as e:
                st.warning(f'Error during normalization: {str(e)}')
    
    ### Handling Missing Data ###

    # Option 1: Drop missing values
    if st.sidebar.checkbox('Drop Missing Values'):
        df_cleaned = drop_missing_values(df_cleaned)
        st.subheader('Dataset with Missing Values Dropped')
        st.write(df_cleaned)
        update_cleaned_df(df_cleaned)

    # Option 2: Fill missing values with mean, median, mode or custom value
    if st.sidebar.checkbox('Fill Missing Values'):
        fill_columns = st.sidebar.multiselect('Select columns to fill missing values:', df_cleaned.columns)

        # Check if any columns were selected
        if not fill_columns:
            st.warning('No columns selected for Filling Missing Values.')
        else:
            try:
                for col in fill_columns:
                    fill_option = st.sidebar.selectbox('Fill missing values with:', ['Mean', 'Median', 'Mode','Custom Value'])
                    if fill_option == 'Mean':
                        df_cleaned[col] = mean(df_cleaned, col)
                    elif fill_option == 'Median':
                        df_cleaned[col] = median(df_cleaned, col)
                    elif fill_option == 'Mode':
                        try:
                            for col2 in fill_columns:
                                df_cleaned[col2] = mode(df_cleaned, col2)
                        except Exception as e:
                            st.warning(f"Error in Mode: {str(e)}")
                    else:
                        custom_value = st.sidebar.text_input('Enter custom value for missing data')
                        df_cleaned[col] = custom(df_cleaned, col)
            except:
                st.warning('Error in Filling Missing Values.')

        st.subheader('Dataset after Filling Missing Values')
        st.write(df_cleaned)
        update_cleaned_df(df_cleaned)

    ### Removing Duplicates ###

    if st.sidebar.checkbox('Remove Duplicates'):
        df_cleaned = drop_duplicates(df_cleaned)
        st.subheader('Dataset after Removing Duplicates')
        st.write(df_cleaned)
        update_cleaned_df(df_cleaned)

    ### Removing Outliers ###

    if st.sidebar.checkbox('Remove Outliers'):
        outlier_columns = st.sidebar.multiselect('Select columns to remove outliers:', df_cleaned.columns)

        if not outlier_columns:
            st.warning('No columns selected for Removing Outliers.')
        else:
            try:
                for col in outlier_columns:
                    df_cleaned = outliers(df_cleaned, col)
                
                st.subheader('Dataset after Removing Outliers')
                st.write(df_cleaned)
                update_cleaned_df(df_cleaned)

            except:
                st.warning('Error in Removing Outliers.')


    ### Linear Regression ###

    if st.sidebar.checkbox('Linear Regression'):
        reg_cols = st.sidebar.multiselect('Select Columns for Linear Regression', df_cleaned.columns)

        if len(reg_cols) < 3:
            st.warning('Please select at least 3 columns for Linear Regression.')
        else:
            try:
                model = fit_linear_regression(df_cleaned, reg_cols[:3])  # Use first three selected columns
                st.success('Linear Regression Model Trained')
            except Exception as e:
                st.warning(f'Error in training model: {e}')

            # Get values for prediction
            values = st.sidebar.text_input('Enter values to predict (comma-separated)')

            if values:
                try:
                    value_arr = [[float(i) for i in values.split(',')]]
                    predicted_value = model.predict(value_arr)
                    st.write(f'Values entered: {value_arr},\nPredicted: {predicted_value[0]}')
                except ValueError:
                    st.warning('Enter values in a valid comma-separated format.')

                # Update the cleaned DataFrame with predictions
                try:
                    st.subheader("Dataset with Predictions:")
                    df_cleaned[f'Linear Predicted {reg_cols[2]}'] = model.predict(df_cleaned[reg_cols[:2]])
                    st.write(df_cleaned)
                    update_cleaned_df(df_cleaned)
                except Exception as e:
                    st.warning(f'Error in Prediction: {e}')

    ### Polynomial Regression ###

    if st.sidebar.checkbox('Polynomial Regression'):
        reg_cols = st.sidebar.multiselect('Select Columns for Polynomial Regression', df_cleaned.columns)

        if not reg_cols:
            st.warning('No columns selected for Polynomial Regression.')

        else:
            try:
                degree = st.sidebar.number_input('Enter degree of polynomial: ', min_value=1, max_value=5, step=1)
            except:
                st.warning('Enter degree of polynomial')
            try:
                model, poly, var1_poly = fit_poly_regression(df_cleaned, reg_cols, degree)
                st.success('Polynomial Regression Model Trained')
            except:
                st.warning('Column Range Error')
            
            try:
                values = st.sidebar.chat_input('Enter values to predict')
                value_arr = [[float(i) for i in values.split(',')]]
                st.write('Values entered: ', values, ',\nPredicted: ', model.predict(poly.transform(value_arr)))
            except:
                st.warning('Enter values in comma separated format')
            try:
                df_cleaned[f'Poly Predicted {reg_cols[0]}, {reg_cols[1]}'] = model.predict(var1_poly)
                st.write(df_cleaned)
                update_cleaned_df(df_cleaned)
            except:
                st.warning('Error in Prediction')
            
    if st.sidebar.checkbox('K-NN Classifier'):
        reg_cols = st.sidebar.multiselect('Select Columns for K-NN Classifier (Last Column as Target)', df_cleaned.columns)

        if not reg_cols:
            st.warning('No columns selected for K-NN Classifier.')
        elif len(reg_cols) < 2:
            st.warning('Please select at least one feature column and one target column.')
        else:
            try:
                # Assume the last selected column is the target
                target_col = reg_cols[-1]
                feature_cols = reg_cols[:-1]

                # # Convert feature columns to numeric, coercing errors to NaN
                # for i in feature_cols:
                #     df_cleaned[i] = df_cleaned[i].apply(pd.to_numeric, errors='coerce')

                # # Drop rows with NaN values in feature columns
                # df_cleaned.dropna(subset=feature_cols, inplace=True)

                neighbors = st.sidebar.number_input('Enter number of nearest neighbors:', min_value=1, step=1)
                knn, features, target = fit_knn(df_cleaned, feature_cols, target_col, neighbors)
                st.success('K-NN Classifier Model Trained')

            except Exception as e:
                st.warning(f'Error training the model: {e}')

            try:
                values = st.sidebar.text_input('Enter values to predict (comma separated):')

                if values:
                    value_arr = [[float(i.strip()) for i in values.split(',')]]  # Stripping whitespace
                    
                    if len(value_arr[0]) != len(feature_cols):
                        st.warning(f'Please enter exactly {len(feature_cols)} values.')
                    else:
                        prediction = knn.predict(value_arr)
                        st.write(f'Values entered: {values},\nPredicted: {prediction[0]}')

            except ValueError as ve:
                st.warning(f'Value error: {ve}')
            except Exception as e:
                st.warning(f'Error in prediction: {e}')

            try:
                df_cleaned[f'KNN Predicted {target_col}'] = knn.predict(df_cleaned[feature_cols])
                st.write(df_cleaned)
                update_cleaned_df(df_cleaned)
            except Exception as e:
                st.warning(f'Error in Prediction: {e}')



    ### Reset button logic
    if st.sidebar.button('Reset', type='primary'):
        df_cleaned = org_df
        st.subheader('Reverting dataset to original state')
        st.write(df_cleaned)
        update_cleaned_df(df_cleaned)

    ### Download Cleaned Dataset ###

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    if st.session_state.cleaned_df is not None:
        st.subheader('Download Cleaned Dataset')
        csv = convert_df(st.session_state.cleaned_df)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name='cleaned_dataset.csv',
            mime='text/csv'
        )