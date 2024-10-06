<h1 align="center">Interactive Dataset Cleaner</h1>

<p align="center"><strong>This project provides an interactive web application for cleaning and analyzing datasets. Built with Streamlit and Python, it allows users to upload datasets in various formats (CSV, Excel, JSON) and perform numerous data cleaning tasks through a user-friendly interface.</strong></p>
<div style="display: flex; justify-content: center; align-items:center; width: 100%;">
<img src="https://github.com/user-attachments/assets/fb5f02b9-eaac-442e-a1b7-bab0739fb7ee" alt="Screenshot of Query: Do One-Hot Encoding on Gender Column" align="center">
</div>


## Features

1. **Dataset Uploading**:
   - Users can upload datasets in CSV, Excel, or JSON formats.

2. **Natural Language Querying**:
   - Users can enter natural language queries to retrieve specific information from the dataset.

3. **Data Cleaning Operations**:
   - Standardization of text columns (lowercase, uppercase, title case).
   - Date format standardization.
   - Rounding of numeric values.
   - Handling missing values (drop, fill with mean/median/mode, or custom value).
   - Removing duplicates.
   - Removing outliers based on the IQR method.
   
4. **Data Transformation**:
   - One-hot encoding for categorical variables.
   - Normalization of numeric columns.

5. **Model Training**:
   - Linear Regression and Polynomial Regression modeling.
   - K-Nearest Neighbors (K-NN) Classifier for predictive modeling.

6. **Visualization**:
   - Users can view the raw and cleaned datasets in real-time.

7. **Download Option**:
   - Users can download the cleaned dataset in CSV format.

## Installation

To run this project locally, make sure you have Python installed along with the required packages. You can install the necessary dependencies using pip:

```bash
pip install streamlit pandas scikit-learn
```

## Usage

1. **Run the App**:
   After installing the required packages, navigate to the project directory and run:

   ```bash
   streamlit run app.py
   ```

2. **Upload Dataset**:
   Use the file uploader to select and upload your dataset.

3. **Perform Data Cleaning**:
   - Use the sidebar to select different data cleaning options and transformations.
   - Enter your natural language queries in the designated input box to filter the dataset.
   
4. **Model Training**:
   - Select features for training the model and enter necessary parameters to get predictions.

5. **Download Cleaned Dataset**:
   Once cleaning and analysis are complete, you can download the cleaned dataset in CSV format.

## Code Structure

### `app.py`
This file contains the main Streamlit application logic. Key sections include:
- File upload handling
- Display of raw and cleaned datasets
- User input handling for natural language queries and data cleaning options
- Model training and prediction functionalities

### `functions.py`
This module contains various data processing and cleaning functions. Key functionalities include:
- Standardization functions for text and numeric data
- Functions for handling missing values and duplicates
- Data transformation functions, including one-hot encoding and normalization
- Model fitting functions for linear regression, polynomial regression, and K-NN classification

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request with your enhancements or bug fixes.

## License

This project is open-source and available under the MIT License.
