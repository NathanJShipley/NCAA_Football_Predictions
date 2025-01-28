import pandas as pd  # data manipulation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Small function to read in the csv data
def read_and_process_data(train_path: str, test_path: str, pos_team_dum_code: bool) -> tuple:
    """
    Reads and processes training and test data from CSV files.

    Parameters:
    train_path (str): Path to the training data CSV file.
    test_path (str): Path to the test data CSV file.
    pos_team_dum_code (bool): Whether to dummy code the 'pos_team' column.

    Returns:
    tuple: A tuple containing the preprocessed training and test data as:
        - x_train (DataFrame): Processed training features.
        - y_train (Series): Target variable for training data.
        - x_test (DataFrame): Processed test features.
        - y_test (Series): Target variable for test data.
    """
    # set arg paths to train and test df
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Split into x and y train and test sets, drop log votes and analytical points from x data.
    # Also drop the conference if present
    # Going to drop these columns from the data
    columns_to_drop = ['points_scored', 'opponent', 'conference']

    # Dynamically drop columns if they exist
    x_train = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns])
    y_train = train_df['points_scored']

    x_test = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns])
    y_test = test_df['points_scored']

    # Save the original 'pos_team' and 'opponent' columns for later
    train_original_test_columns = train_df[['pos_team', 'opponent', 'year', 'week']]
    train_original_test_columns = train_original_test_columns.rename(columns={'year':'original_year', 'week':'original_week'})

    test_original_test_columns = test_df[['pos_team', 'opponent', 'year', 'week']]
    test_original_test_columns = test_original_test_columns.rename(columns={'year':'original_year', 'week':'original_week'})

    # Now going to scale the x data, starting with training data first and then apply to test
    columns_to_scale = x_train.drop(columns=['pos_team']).columns

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the training data
    x_train[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale])

    # Transform the test data using the parameters from training
    x_test[columns_to_scale] = scaler.transform(x_test[columns_to_scale])

    # Handle dummy coding of 'pos_team' if required
    if pos_team_dum_code:
        # Dummy code and convert boolean dummies to integers
        x_train_encoded = pd.get_dummies(x_train, columns=['pos_team'], drop_first=True).astype(int)
        x_test_encoded = pd.get_dummies(x_test, columns=['pos_team'], drop_first=True).astype(int)
    else:
        # Keep the original data without dummy coding
        x_train_encoded = x_train.copy()
        x_test_encoded = x_test.copy()

    # Some teams are missing from the test data from the train data, so let add them back real quick to the test data
    # Add missing columns to x_test_encoded in a single operation
    missing_cols = set(x_train_encoded.columns) - set(x_test_encoded.columns)

    for cols in missing_cols:
        x_test_encoded[cols] = 0

    # Re-align columns in x_test_encoded to match the order of x_train_encoded
    x_test_encoded = x_test_encoded[x_train_encoded.columns]

    # Convert all boolean columns to integers (True -> 1, False -> 0) using the 'astype' method for the entire dataframe
    bool_cols = x_train_encoded.select_dtypes(include=['bool']).columns  # Identify boolean columns
    x_train_encoded[bool_cols] = x_train_encoded[bool_cols].astype(int)  # Convert them to int
    x_test_encoded[bool_cols] = x_test_encoded[bool_cols].astype(int)

    # Verify the data types to ensure all booleans are now integers
    print("Data types for x_train after conversion:", x_train_encoded.dtypes.value_counts())
    print("Data types for x_test after conversion:", x_test_encoded.dtypes.value_counts())


    # Check this when building out the model
    print("train data has shape:", x_train_encoded.shape)  # Expected output should be (num_samples, 18874 samples and 503 columns
    print("test data has shape:", x_test_encoded.shape)  # Expected output should be (num_samples, 134 samples and 503 columns, so good match there)

    return x_train_encoded, y_train, x_test_encoded, y_test, train_original_test_columns, test_original_test_columns


# Example of what the script is looking for and how to read in the data
# Read in the csv data. Note that the test data is based on the most recent week of data
#
# x_train, y_train, x_test, y_test = read_and_process_data(
#     r"E:\github_repos\Private_Projects\NCAA_FBS_AP_Ranking_Predictions\python_ap\scripts_and_data\data\full_train_data.csv",
#     r"E:\github_repos\Private_Projects\NCAA_FBS_AP_Ranking_Predictions\python_ap\scripts_and_data\data\full_test_data.csv",
#     True
# )

# Lets make a model evaluation function as well that I can use in my other models
def model_evaluation(y_train_test, y_pred):
    # fit metrics
    r2 = r2_score(y_train_test, y_pred)
    mse = mean_squared_error(y_train_test, y_pred)
    mae = mean_absolute_error(y_train_test, y_pred)

    # print results
    print(f"R-squared: {r2:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")