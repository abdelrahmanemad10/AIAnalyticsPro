import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user

def get_openai_client():
    """
    Initialize and return an OpenAI client
    
    Returns:
        OpenAI: OpenAI client object
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
    
    return OpenAI(api_key=api_key)

def analyze_dataset(df, analysis_type="general"):
    """
    Analyze the dataset using OpenAI API
    
    Args:
        df: pandas DataFrame
        analysis_type: Type of analysis to perform (general, correlations, outliers, trends)
        
    Returns:
        dict: Dictionary containing analysis results
    """
    client = get_openai_client()
    if client is None:
        return {"error": "OpenAI client initialization failed"}
    
    # Prepare dataset information
    column_info = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        unique_count = df[col].nunique()
        missing_count = df[col].isna().sum()
        
        column_info.append({
            "name": col,
            "type": col_type,
            "unique_values": int(unique_count),
            "missing_values": int(missing_count),
        })
    
    # Add sample data (first 5 rows)
    sample_data = df.head(5).to_dict(orient="records")
    
    # Prepare statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    statistics = {}
    if numeric_cols:
        statistics["numeric"] = df[numeric_cols].describe().to_dict()
    
    # Create a prompt based on analysis type
    if analysis_type == "general":
        prompt = """
        As a data analysis expert, analyze this dataset and provide insights. Include:
        1. A summary of the dataset's structure and content
        2. Key patterns or trends observed
        3. Potential data quality issues
        4. Recommended visualizations
        5. Suggested business insights that could be explored further
        
        Format your response as JSON with these sections.
        """
    elif analysis_type == "correlations":
        # Include correlation matrix for numeric columns
        corr_matrix = df[numeric_cols].corr().to_dict() if numeric_cols else {}
        prompt = """
        Analyze the correlations in this dataset and provide insights. Include:
        1. Strongest positive and negative correlations
        2. Explanation of what each significant correlation might mean
        3. Recommendations for visualizations to explore these relationships
        4. Potential causation hypotheses (while noting correlation ≠ causation)
        
        Format your response as JSON with these sections.
        """
        statistics["correlations"] = corr_matrix
    elif analysis_type == "outliers":
        prompt = """
        Identify and analyze potential outliers in this dataset. Include:
        1. Which columns contain potential outliers
        2. Recommended methods for outlier detection for this data
        3. Potential impact of outliers on analysis
        4. Suggestions for handling outliers
        
        Format your response as JSON with these sections.
        """
    elif analysis_type == "trends":
        prompt = """
        Analyze this dataset for trends and patterns. Include:
        1. Key trends or patterns observed in the data
        2. Seasonal patterns if time data is present
        3. Group differences or segments in the data
        4. Recommended visualizations to explore these trends
        
        Format your response as JSON with these sections.
        """
    
    # Create the dataset info as a JSON string
    dataset_info = {
        "columns": column_info,
        "sample_data": sample_data,
        "statistics": statistics,
        "row_count": len(df),
        "column_count": len(df.columns)
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert data analyst providing insights on datasets. Return your analysis in valid JSON format."},
                {"role": "user", "content": prompt + "\n\nHere is the dataset information:\n" + json.dumps(dataset_info, default=str)}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        return {"error": f"Error during analysis: {str(e)}"}

def get_visualization_recommendations(df):
    """
    Get AI-powered recommendations for visualizations
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: List of visualization recommendations
    """
    client = get_openai_client()
    if client is None:
        return [{"error": "OpenAI client initialization failed"}]
    
    # Prepare column information
    column_info = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        unique_count = df[col].nunique()
        missing_count = df[col].isna().sum()
        
        # For numeric columns, include basic stats
        if np.issubdtype(df[col].dtype, np.number):
            stats = {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
            }
        else:
            stats = {}
        
        column_info.append({
            "name": col,
            "type": col_type,
            "unique_values": int(unique_count),
            "missing_values": int(missing_count),
            "stats": stats
        })
    
    # Create the prompt
    prompt = """
    As a data visualization expert, recommend the most insightful visualizations for this dataset.
    
    For each recommended visualization:
    1. Specify the chart type (bar, line, scatter, pie, box, heatmap, etc.)
    2. Which columns to use for x-axis, y-axis, and any other dimensions like color or size
    3. Provide a title for the visualization
    4. Explain why this visualization would be insightful
    5. Suggest any transformations needed for the data
    
    Return exactly 5 recommendations, sorted by potential insight value.
    Format your response as a JSON array of visualization objects with these fields:
    - chart_type: The type of chart
    - x_column: Column for the x-axis
    - y_column: Column for the y-axis (if applicable)
    - color_column: Column for color differentiation (if applicable)
    - title: Suggested title
    - description: Why this visualization is insightful
    - preparation: Any data preparation needed
    """
    
    # Dataset info
    dataset_info = {
        "columns": column_info,
        "row_count": len(df),
        "column_count": len(df.columns)
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert data visualization advisor. Return your recommendations in valid JSON format."},
                {"role": "user", "content": prompt + "\n\nHere is the dataset information:\n" + json.dumps(dataset_info, default=str)}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure the result is a list
        if isinstance(result, dict) and "recommendations" in result:
            return result["recommendations"]
        elif isinstance(result, list):
            return result
        else:
            return [{"error": "Unexpected response format", "raw_response": result}]
    
    except Exception as e:
        return [{"error": f"Error getting visualization recommendations: {str(e)}"}]

def generate_natural_language_query(df, query):
    """
    Process a natural language query about the dataset using OpenAI
    
    Args:
        df: pandas DataFrame
        query: Natural language query string
        
    Returns:
        dict: Results of the query
    """
    client = get_openai_client()
    if client is None:
        return {"error": "OpenAI client initialization failed"}
    
    # Prepare dataset information
    column_info = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        unique_values = df[col].nunique()
        example_values = df[col].dropna().sample(min(5, max(1, unique_values))).tolist()
        
        column_info.append({
            "name": col,
            "type": col_type,
            "unique_count": int(unique_values),
            "example_values": [str(val) for val in example_values]
        })
    
    # Create the dataset info
    dataset_info = {
        "columns": column_info,
        "row_count": len(df),
        "column_count": len(df.columns)
    }
    
    # First, generate Python code to answer the query
    prompt_for_code = f"""
    Given this dataset information and the user's query: "{query}", 
    generate Python code using pandas to answer this query.
    
    The pandas DataFrame is called 'df'. Generate only the code to compute the answer, not to create or load the DataFrame.
    Return the code in a JSON object with a single key 'code'. Do not include any explanations or markdown formatting.
    
    Keep the code concise, efficient, and focused only on answering the question. Assume the DataFrame is already loaded.
    Include proper error handling if the query can't be answered with the available data.
    """
    
    try:
        # Get the code to answer the query
        code_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in pandas data analysis. Generate Python code to answer the user's question about a dataset."},
                {"role": "user", "content": prompt_for_code + "\n\nHere is the dataset information:\n" + json.dumps(dataset_info, default=str)}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        code_result = json.loads(code_response.choices[0].message.content)
        code = code_result.get("code", "")
        
        # Execute the code in a safe environment
        try:
            # Prepare the local environment
            local_env = {"df": df, "pd": pd, "np": np}
            
            # Execute the code
            exec_result = {}
            exec(f"result = {code}", local_env, exec_result)
            answer = exec_result.get("result")
            
            # Convert to serializable format
            if isinstance(answer, pd.DataFrame):
                answer = answer.to_dict(orient="records")
            elif isinstance(answer, pd.Series):
                answer = answer.to_dict()
            elif isinstance(answer, np.ndarray):
                answer = answer.tolist()
            
            # Get an explanation of the result
            explanation_prompt = f"""
            Given this dataset information, the user's query: "{query}", and the code:
            ```python
            {code}
            ```
            
            Explain in simple terms:
            1. What the code does 
            2. How to interpret the result
            3. Any caveats or limitations to be aware of
            
            Return your explanation in a JSON object with key 'explanation'.
            """
            
            explanation_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst explaining analysis results to non-technical users."},
                    {"role": "user", "content": explanation_prompt + "\n\nHere is the dataset information:\n" + json.dumps(dataset_info, default=str)}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            explanation_result = json.loads(explanation_response.choices[0].message.content)
            explanation = explanation_result.get("explanation", "")
            
            return {
                "query": query,
                "code": code,
                "result": answer,
                "explanation": explanation,
                "success": True
            }
            
        except Exception as exec_error:
            return {
                "query": query,
                "code": code,
                "error": f"Error executing code: {str(exec_error)}",
                "success": False
            }
    
    except Exception as e:
        return {
            "query": query,
            "error": f"Error processing query: {str(e)}",
            "success": False
        }

def suggest_data_transformations(df):
    """
    Suggest data transformations to improve analysis
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: List of suggested transformations
    """
    client = get_openai_client()
    if client is None:
        return [{"error": "OpenAI client initialization failed"}]
    
    # Prepare column information
    column_info = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        unique_count = df[col].nunique()
        missing_count = df[col].isna().sum()
        
        column_info.append({
            "name": col,
            "type": col_type,
            "unique_values": int(unique_count),
            "missing_values": int(missing_count),
            "missing_percentage": round(100 * missing_count / len(df), 2) if len(df) > 0 else 0
        })
    
    # Create the prompt
    prompt = """
    As a data preprocessing expert, suggest transformations to improve this dataset for analysis.
    Consider:
    
    1. Handling missing values
    2. Feature engineering opportunities
    3. Data type conversions
    4. Outlier handling
    5. Normalization or scaling needs
    6. Date/time field processing
    7. Categorical encoding
    8. Aggregations that might be useful
    
    For each suggestion, provide:
    - The transformation type
    - Which column(s) it applies to
    - The rationale
    - A code snippet showing how to implement it in pandas
    
    Return your suggestions as a JSON array of transformation objects.
    """
    
    # Dataset info
    dataset_info = {
        "columns": column_info,
        "row_count": len(df),
        "column_count": len(df.columns),
        "sample_data": df.head(5).to_dict(orient="records")
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert data preprocessing advisor. Return your suggestions in valid JSON format."},
                {"role": "user", "content": prompt + "\n\nHere is the dataset information:\n" + json.dumps(dataset_info, default=str)}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure the result is a list
        if isinstance(result, dict) and "transformations" in result:
            return result["transformations"]
        elif isinstance(result, list):
            return result
        else:
            return [{"error": "Unexpected response format", "raw_response": result}]
    
    except Exception as e:
        return [{"error": f"Error generating transformation suggestions: {str(e)}"}]
