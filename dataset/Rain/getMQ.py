from numpy import mean
import pandas as pd

def calculate_Q_M(tp_series, date_series, original_df):
    """
    Calculate Q (inter-demand intervals) and M (demand sizes) while preserving all original data columns.
    
    Parameters:
    tp_series (list or pd.Series): Time series data (tp column)
    date_series (list or pd.Series): Corresponding date series
    original_df (pd.DataFrame): Original dataframe containing all columns
    
    Returns:
    pd.DataFrame: DataFrame containing all original columns plus Q and M values
                  only for rows where demand occurred (tp != 1e-5)
    """
    result_data = []
    last_nonzero_index = -1
    
    for current_index, (tp_value, date_value) in enumerate(zip(tp_series, date_series)):
        if tp_value != 1e-5:  # Check for non-zero demand
            # Calculate inter-demand interval
            q_value = current_index - last_nonzero_index
            
            # Get all original columns for this row
            row_data = original_df.iloc[current_index].to_dict()
            
            # Add calculated Q and M values
            row_data['Q'] = q_value
            row_data['M'] = tp_value
            
            result_data.append(row_data)
            
            # Update last non-zero index
            last_nonzero_index = current_index
            
    return pd.DataFrame(result_data)

# Read CSV file
try:
    csv_file_path = 'JFNG_data_15min.csv'
    df = pd.read_csv(csv_file_path)
    
    if 'tp' not in df.columns:
        print(f"Error: 'tp' column not found in '{csv_file_path}'")
    else:
        # Calculate Q and M while preserving all columns
        result_df = calculate_Q_M(df['tp'], df['date'], df)
        
        # Print statistics
        print("\nDemand Interval (Q) Statistics:")
        print(f"MAX Q: {result_df['Q'].max()}")
        print(f"MIN Q: {result_df['Q'].min()}")
        print(f"MEAN Q: {result_df['Q'].mean():.2f}")
        print(f"Count: {len(result_df['Q'])}")
        
        print("\nDemand Size (M) Statistics:")
        print(f"MAX M: {result_df['M'].max()}")
        print(f"MIN M: {result_df['M'].min()}")
        print(f"MEAN M: {result_df['M'].mean():.2f}")
        print(f"Count: {len(result_df['M'])}")
        
        # Save results with all original columns
        result_df.to_csv('output_Q_M_full.csv', index=False)
        print("\nResults saved to 'output_Q_M_full.csv' with all original columns")

except FileNotFoundError:
    print(f"Error: File '{csv_file_path}' not found")
except Exception as e:
    print(f"Error: {str(e)}")