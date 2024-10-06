import re
from transformers import pipeline
from fuzzywuzzy import process
from functions import *

# Initialize zero-shot classification pipeline
nlp_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible intents
# Example intent keywords for better distinction
intents = {
    'fill_missing_mean': ['fill with mean', 'mean', 'average'],
    'fill_missing_median': ['fill with median', 'median', 'middle', 'mid'],
    'fill_missing_mode': ['fill with mode', 'mode', 'recurrent', 'most frequent', 'most common'],
    'fill_missing_custom': ['fill with custom', 'custom', 'personal'],
    'drop_missing': ['drop', 'missing', 'null', 'nan'],
    'apply_normalization': ['normalize', 'normalization', 'scaling'],
    'one_hot_encoding': ['one-hot', 'encoding', 'encode'],
    'remove_duplicates': ['duplicate', 'duplicates', 'drop'],
    'remove_outliers': ['outlier', 'outliers', 'remove'],
    'Capitalize': ['capitalize', 'capital', 'uppercase'],
    'lowercase': ['lowercase', 'lower'],
    'uppercase': ['uppercase', 'upper'],
    'Linear Regression': ['linear regression', 'linear', 'regression'],
    'Polynomial Regression': ['polynomial regression', 'polynomial'],
    'KNN': ['knn', 'k-nearest neighbors', 'neighbors']
}


def classify_intent_with_columns_and_values(text, columns):
    """
    Classify the intent, columns, and extract any target values from the query.
    """
    if not text or text.strip() == "":
        print("Input text is empty or invalid.")
        return None, None, None
    
    # Generate intent-column combinations
    intent_column_combinations = [f"{intent} on {col}" for intent in intents for col in columns]
    
    try:
        # Classify the text using zero-shot classification
        result = nlp_pipeline(text, intent_column_combinations)
        
        # Get the top result (intent + column)
        top_match = result['labels'][0]
        
        # Split the intent and column
        if " on " in top_match:
            intent, first_column = top_match.split(" on ")
        else:
            print("Error in parsing the result format.")
            return None, None, None
        
        # Refine the intent
        refined_intent = refine_intent_classification(intent, text)
        
        # Find all mentioned columns in the text
        mentioned_columns = [col for col in columns if col in text]

        # Allow for comma-separated column names in the text
        if len(mentioned_columns) == 0 and "," in text:
            column_candidates = text.split(",")
            mentioned_columns = [col.strip() for col in column_candidates if col.strip() in columns]

        if not mentioned_columns:
            mentioned_columns = [first_column]  # Default to first detected column if none explicitly mentioned
        
        # Extract numerical or target values from the text
        values = extract_values_from_text(text)

        print(f"Identified intent: {refined_intent}")
        print(f"Identified columns: {mentioned_columns}")
        print(f"Identified values: {values}")
        
        return refined_intent, mentioned_columns, values
    
    except Exception as e:
        print(f"Error during classification: {e}")
        return None, None, None

def refine_intent_classification(intent, text):
    for refined_intent, keywords in intents.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                return refined_intent
    return intent  # return original intent if no refinement is made

def extract_column(text, df):
    """Extract the best matching column from the dataset based on the user's text input."""
    column_names = df.columns.tolist()
    match, score = process.extractOne(text, column_names)
    
    if score > 70:  # Threshold for a good match
        return match
    else:
        return None  # No good match found


def extract_values_from_text(text):
    """
    Extract numeric values or other parameters from the query.
    """
    # Extract both integers and floats using a regex
    values = re.findall(r'\d+\.?\d*', text)
    
    if values:
        return [float(value) for value in values]  # Convert to floats for further use
    return []

def execute_intent(df_cleaned, intent, column, values):
    """
    Execute the intent based on the user's input.
    """
    if intent == 'fill_missing_mean':
        return mean(df_cleaned, column)
    elif intent == 'fill_missing_median':
        return median(df_cleaned, column)
    elif intent == 'fill_missing_mode':
        return mode(df_cleaned, column)
    elif intent == 'fill_missing_custom':
        return custom(df_cleaned, column)
    elif intent == 'drop_missing':
        return drop_missing_values(df_cleaned)
    elif intent == 'apply_normalization':
        return normalize(df_cleaned, column)
    elif intent == 'one_hot_encoding':
        return One_hot_encoding(df_cleaned, column)
    elif intent == 'remove_duplicates':
        return drop_duplicates(df_cleaned)
    elif intent == 'remove_outliers':
        return outliers(df_cleaned, column)
    elif intent == 'capitalize' or intent == 'title':
        return title(df_cleaned, column)
    elif intent == 'lowercase':
        return lower(df_cleaned, column)
    elif intent == 'uppercase':
        return upper(df_cleaned, column)
    elif intent == 'linear_regression':
        return lin_regressor(df_cleaned, column, values)
    elif intent == 'polynomial_regression':
        return poly_regressor(df_cleaned, column, values)
    elif intent == 'knn':
        return fit_knn(df_cleaned, column, values)