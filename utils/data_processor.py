import pandas as pd
import numpy as np
import io
import streamlit as st
from io import StringIO

def process_uploaded_file(uploaded_file):
    """
    Process the uploaded file based on its type and return a pandas DataFrame
    
    Args:
        uploaded_file: Streamlit's UploadedFile object
        
    Returns:
        tuple: (pandas DataFrame, filename)
    """
    filename = uploaded_file.name
    file_extension = filename.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            # Try different encodings and separators for CSV files
            try:
                data = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # If UTF-8 fails, try another common encoding
                data = pd.read_csv(uploaded_file, encoding='latin1')
            except pd.errors.ParserError:
                # If comma separator fails, try semicolon 
                data = pd.read_csv(uploaded_file, sep=';')
                
        elif file_extension in ['xlsx', 'xls']:
            data = pd.read_excel(uploaded_file)
            
        elif file_extension == 'json':
            data = pd.read_json(uploaded_file)
            
        elif file_extension == 'tsv':
            data = pd.read_csv(uploaded_file, sep='\t')
            
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Basic data cleaning
        data = clean_column_names(data)
        
        return data, filename
    
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

def clean_column_names(df):
    """
    Clean column names: lowercase, replace spaces with underscores, 
    and remove special characters
    
    Args:
        df: pandas DataFrame
        
    Returns:
        pandas DataFrame with cleaned column names
    """
    df.columns = [str(col).lower().replace(' ', '_').replace('.', '_') for col in df.columns]
    return df

def load_sample_data():
    """
    Load sample data for demonstration purposes
    
    Returns:
        tuple: (pandas DataFrame, filename)
    """
    # Sample sales data with various column types
    data = {
        'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'product_id': np.random.randint(1000, 9999, size=100),
        'product_name': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'], size=100),
        'category': np.random.choice(['Electronics', 'Accessories', 'Software'], size=100),
        'price': np.random.uniform(100, 2000, size=100).round(2),
        'quantity': np.random.randint(1, 10, size=100),
        'customer_id': np.random.randint(10000, 99999, size=100),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], size=100),
        'is_promotion': np.random.choice([True, False], size=100),
        'rating': np.random.randint(1, 6, size=100)
    }
    
    # Calculate total sales
    df = pd.DataFrame(data)
    df['total_sales'] = df['price'] * df['quantity']
    
    # Add some missing values for realism
    mask = np.random.random(df.shape) < 0.05
    df = df.mask(mask)
    
    return df, "sample_sales_data.csv"

def filter_dataframe(df, filters):
    """
    Filter a DataFrame based on the provided filters
    
    Args:
        df: pandas DataFrame
        filters: dict of column:value pairs to filter on
        
    Returns:
        filtered pandas DataFrame
    """
    filtered_df = df.copy()
    
    for column, value in filters.items():
        if column in filtered_df.columns:
            if value is not None:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                elif isinstance(value, tuple) and len(value) == 2:
                    # Range filter (min, max)
                    min_val, max_val = value
                    if min_val is not None:
                        filtered_df = filtered_df[filtered_df[column] >= min_val]
                    if max_val is not None:
                        filtered_df = filtered_df[filtered_df[column] <= max_val]
                else:
                    # Exact match
                    filtered_df = filtered_df[filtered_df[column] == value]
    
    return filtered_df

def get_column_stats(df):
    """
    Get basic statistics for each column in the DataFrame
    
    Args:
        df: pandas DataFrame
        
    Returns:
        dict: Dictionary with column statistics
    """
    stats = {}
    
    # Numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        stats['numeric'] = df[numeric_cols].describe().to_dict()
        
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    if cat_cols:
        stats['categorical'] = {}
        for col in cat_cols:
            value_counts = df[col].value_counts().head(10).to_dict()
            unique_count = df[col].nunique()
            missing_count = df[col].isna().sum()
            stats['categorical'][col] = {
                'value_counts': value_counts,
                'unique_count': unique_count,
                'missing_count': missing_count
            }
    
    # Date columns
    date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    if date_cols:
        stats['datetime'] = {}
        for col in date_cols:
            stats['datetime'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'missing_count': df[col].isna().sum()
            }
    
    return stats

def generate_data_summary(df):
    """
    Generate a text summary of the dataset
    
    Args:
        df: pandas DataFrame
        
    Returns:
        str: Text summary of the dataset
    """
    summary = []
    summary.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        summary.append(f"There are {len(missing_cols)} columns with missing values:")
        for col, count in missing_cols.items():
            percentage = round((count / len(df)) * 100, 2)
            summary.append(f"  - {col}: {count} missing values ({percentage}%)")
    else:
        summary.append("There are no missing values in the dataset.")
    
    # Data types
    summary.append("\nData types:")
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns.tolist()
        summary.append(f"  - {dtype}: {len(cols)} columns")
        
    # Numerical columns summary
    num_cols = df.select_dtypes(include=['number']).columns
    if not num_cols.empty:
        summary.append("\nNumerical columns summary:")
        for col in num_cols[:5]:  # List first 5 to avoid overwhelming
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            summary.append(f"  - {col}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")
        if len(num_cols) > 5:
            summary.append(f"  - (and {len(num_cols) - 5} more numerical columns)")
    
    return "\n".join(summary)

def transform_data(df, transformations):
    """
    Apply transformations to the DataFrame
    
    Args:
        df: pandas DataFrame
        transformations: list of transformation operations to apply
        
    Returns:
        transformed pandas DataFrame
    """
    transformed_df = df.copy()
    
    for transform in transformations:
        operation = transform.get('operation')
        
        if operation == 'filter':
            column = transform.get('column')
            condition = transform.get('condition')
            value = transform.get('value')
            
            if condition == 'equals':
                transformed_df = transformed_df[transformed_df[column] == value]
            elif condition == 'not_equals':
                transformed_df = transformed_df[transformed_df[column] != value]
            elif condition == 'greater_than':
                transformed_df = transformed_df[transformed_df[column] > value]
            elif condition == 'less_than':
                transformed_df = transformed_df[transformed_df[column] < value]
            elif condition == 'contains':
                transformed_df = transformed_df[transformed_df[column].astype(str).str.contains(value)]
            elif condition == 'in':
                transformed_df = transformed_df[transformed_df[column].isin(value)]
                
        elif operation == 'sort':
            column = transform.get('column')
            ascending = transform.get('ascending', True)
            transformed_df = transformed_df.sort_values(by=column, ascending=ascending)
            
        elif operation == 'group_by':
            columns = transform.get('columns')
            agg_func = transform.get('aggregation')
            transformed_df = transformed_df.groupby(columns).agg(agg_func).reset_index()
            
        elif operation == 'new_column':
            column = transform.get('column')
            expression = transform.get('expression')
            # Safe eval of the expression
            transformed_df[column] = transformed_df.eval(expression)
            
        elif operation == 'rename':
            mapping = transform.get('mapping')
            transformed_df = transformed_df.rename(columns=mapping)
            
        elif operation == 'drop':
            columns = transform.get('columns')
            transformed_df = transformed_df.drop(columns=columns)
            
    return transformed_df

def df_to_csv(df):
    """
    Convert DataFrame to CSV string
    
    Args:
        df: pandas DataFrame
        
    Returns:
        str: CSV string
    """
    return df.to_csv(index=False)

def df_to_excel(df):
    """
    Convert DataFrame to Excel bytes
    
    Args:
        df: pandas DataFrame
        
    Returns:
        bytes: Excel file bytes
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
    return output.getvalue()
