import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st

def get_recommended_charts(df, column_name):
    """
    Get recommended chart types based on data column characteristics
    
    Args:
        df: pandas DataFrame
        column_name: Name of the column to analyze
        
    Returns:
        list: List of recommended chart types
    """
    recommendations = []
    
    # Check if column exists
    if column_name not in df.columns:
        return []
    
    column_dtype = df[column_name].dtype
    nunique = df[column_name].nunique()
    
    # Numeric column
    if np.issubdtype(column_dtype, np.number):
        recommendations.append({
            'chart_type': 'histogram',
            'title': f'Distribution of {column_name}',
            'description': 'Shows the distribution of values',
            'score': 0.9
        })
        recommendations.append({
            'chart_type': 'box',
            'title': f'Box Plot of {column_name}',
            'description': 'Shows statistical summary with outliers',
            'score': 0.8
        })
        
        # If less than 30 unique values, also recommend bar chart
        if nunique < 30:
            recommendations.append({
                'chart_type': 'bar',
                'title': f'Count of {column_name} Values',
                'description': 'Shows frequency of each value',
                'score': 0.7
            })
    
    # Categorical column
    elif (df[column_name].dtype == 'object' or 
          df[column_name].dtype == 'category' or 
          df[column_name].dtype == 'bool'):
        
        # If many unique values, recommend treemap or pie
        if 2 <= nunique <= 10:
            recommendations.append({
                'chart_type': 'pie',
                'title': f'Distribution of {column_name}',
                'description': 'Shows proportion of each category',
                'score': 0.9
            })
        
        # Bar chart is almost always good for categorical
        recommendations.append({
            'chart_type': 'bar',
            'title': f'Count by {column_name}',
            'description': 'Shows count for each category',
            'score': 0.9
        })
        
        if nunique > 10:
            recommendations.append({
                'chart_type': 'treemap',
                'title': f'Treemap of {column_name}',
                'description': 'Hierarchical view of categories',
                'score': 0.8
            })
    
    # Date/time column
    elif pd.api.types.is_datetime64_dtype(df[column_name]):
        recommendations.append({
            'chart_type': 'line',
            'title': f'Timeline of Records by {column_name}',
            'description': 'Shows trends over time',
            'score': 0.9
        })
        recommendations.append({
            'chart_type': 'bar',
            'title': f'Count by {column_name}',
            'description': 'Shows count for each time period',
            'score': 0.8
        })
    
    return recommendations

def get_bivariate_chart_recommendations(df, x_col, y_col):
    """
    Get recommended charts for relationship between two columns
    
    Args:
        df: pandas DataFrame
        x_col: Name of the first column
        y_col: Name of the second column
        
    Returns:
        list: List of recommended chart types
    """
    recommendations = []
    
    # Check if columns exist
    if x_col not in df.columns or y_col not in df.columns:
        return []
    
    x_dtype = df[x_col].dtype
    y_dtype = df[y_col].dtype
    
    # Both numeric - scatter, heatmap
    if np.issubdtype(x_dtype, np.number) and np.issubdtype(y_dtype, np.number):
        recommendations.append({
            'chart_type': 'scatter',
            'title': f'{x_col} vs {y_col}',
            'description': 'Shows relationship between two numeric variables',
            'score': 0.9
        })
        recommendations.append({
            'chart_type': 'heatmap',
            'title': f'Heatmap of {x_col} vs {y_col}',
            'description': 'Shows density of points',
            'score': 0.7
        })
    
    # Categorical vs Numeric - box plot, bar chart
    elif ((df[x_col].dtype == 'object' or df[x_col].dtype == 'category') and 
          np.issubdtype(y_dtype, np.number)):
        recommendations.append({
            'chart_type': 'box',
            'title': f'Distribution of {y_col} by {x_col}',
            'description': 'Shows distribution for each category',
            'score': 0.9
        })
        recommendations.append({
            'chart_type': 'bar',
            'title': f'Average {y_col} by {x_col}',
            'description': 'Shows average for each category',
            'score': 0.8
        })
    
    # Numeric vs Categorical - box plot, bar chart (reversed)
    elif (np.issubdtype(x_dtype, np.number) and 
          (df[y_col].dtype == 'object' or df[y_col].dtype == 'category')):
        recommendations.append({
            'chart_type': 'box',
            'title': f'Distribution of {x_col} by {y_col}',
            'description': 'Shows distribution for each category',
            'score': 0.9
        })
        recommendations.append({
            'chart_type': 'violin',
            'title': f'Violin plot of {x_col} by {y_col}',
            'description': 'Shows distribution density for each category',
            'score': 0.7
        })
    
    # Both categorical - heatmap, stacked bar
    elif ((df[x_col].dtype == 'object' or df[x_col].dtype == 'category') and 
          (df[y_col].dtype == 'object' or df[y_col].dtype == 'category')):
        recommendations.append({
            'chart_type': 'heatmap',
            'title': f'Relationship between {x_col} and {y_col}',
            'description': 'Shows frequency of combinations',
            'score': 0.9
        })
        recommendations.append({
            'chart_type': 'stacked_bar',
            'title': f'Proportion of {y_col} by {x_col}',
            'description': 'Shows relative proportions',
            'score': 0.8
        })
    
    # Datetime vs Numeric - line chart
    elif pd.api.types.is_datetime64_dtype(df[x_col]) and np.issubdtype(y_dtype, np.number):
        recommendations.append({
            'chart_type': 'line',
            'title': f'{y_col} over Time',
            'description': 'Shows trends over time',
            'score': 0.9
        })
        recommendations.append({
            'chart_type': 'area',
            'title': f'{y_col} Cumulative over Time',
            'description': 'Shows cumulative values over time',
            'score': 0.7
        })
    
    return recommendations

def create_chart(df, chart_type, x_col, y_col=None, color_col=None, title=None, **kwargs):
    """
    Create a chart based on the specified type and columns
    
    Args:
        df: pandas DataFrame
        chart_type: Type of chart to create
        x_col: Column for x-axis
        y_col: Column for y-axis (optional)
        color_col: Column for color differentiation (optional)
        title: Chart title (optional)
        **kwargs: Additional arguments for chart customization
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = None
    
    # Set default title if not provided
    if title is None:
        if y_col:
            title = f"{y_col} by {x_col}"
        else:
            title = f"Analysis of {x_col}"
    
    try:
        if chart_type == 'bar':
            if y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title, **kwargs)
            else:
                # If no y_col, create count plot
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, 'count']
                fig = px.bar(counts, x=x_col, y='count', title=title, **kwargs)
                
        elif chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title, **kwargs)
            
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title, **kwargs)
            
        elif chart_type == 'pie':
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, 'count']
            fig = px.pie(counts, names=x_col, values='count', title=title, **kwargs)
            
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x_col, color=color_col, title=title, **kwargs)
            
        elif chart_type == 'box':
            if y_col:
                fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title, **kwargs)
            else:
                fig = px.box(df, x=x_col, title=title, **kwargs)
                
        elif chart_type == 'violin':
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, title=title, box=True, **kwargs)
            
        elif chart_type == 'heatmap':
            # For heatmap, we need to pivot the data
            if y_col:
                heatmap_data = df.groupby([x_col, y_col]).size().reset_index(name='count')
                heatmap_pivot = heatmap_data.pivot(index=y_col, columns=x_col, values='count')
                fig = px.imshow(heatmap_pivot, title=title, **kwargs)
            else:
                # Correlation heatmap if no y_col specified
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(corr_matrix, title="Correlation Matrix", **kwargs)
                
        elif chart_type == 'area':
            fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title, **kwargs)
            
        elif chart_type == 'treemap':
            if y_col:
                fig = px.treemap(df, path=[x_col], values=y_col, title=title, **kwargs)
            else:
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, 'count']
                fig = px.treemap(counts, path=[x_col], values='count', title=title, **kwargs)
                
        elif chart_type == 'stacked_bar':
            if y_col:
                # Create a cross-tabulation
                cross_tab = pd.crosstab(df[x_col], df[y_col], normalize='index')
                fig = px.bar(cross_tab, title=title, barmode='stack', **kwargs)
            else:
                raise ValueError("y_col is required for stacked bar chart")
                
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Apply common layout settings
        fig.update_layout(
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
    except Exception as e:
        # Return error message as figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
    
    return fig

def get_correlation_matrix(df, include_only=None):
    """
    Generate a correlation matrix for numeric columns
    
    Args:
        df: pandas DataFrame
        include_only: List of column names to include (optional)
        
    Returns:
        pandas DataFrame: Correlation matrix
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Filter columns if specified
    if include_only:
        numeric_df = numeric_df[[col for col in include_only if col in numeric_df.columns]]
    
    # Return correlation matrix
    return numeric_df.corr()

def display_chart_with_controls(df, chart_type, x_col, y_col=None, default_title=None):
    """
    Display a chart with configuration controls
    
    Args:
        df: pandas DataFrame
        chart_type: Type of chart
        x_col: Column for x-axis
        y_col: Column for y-axis (optional)
        default_title: Default chart title (optional)
        
    Returns:
        None
    """
    st.subheader(f"{chart_type.title()} Chart Configuration")
    
    # Title control
    title = st.text_input("Chart Title", value=default_title or f"{chart_type.title()} Chart")
    
    # Color column selector (if applicable)
    color_options = [None] + df.columns.tolist()
    color_col = st.selectbox("Color By", options=color_options, index=0)
    
    # Additional controls based on chart type
    kwargs = {}
    
    if chart_type in ['histogram', 'box']:
        kwargs['nbins'] = st.slider("Number of Bins", min_value=5, max_value=100, value=20, step=5)
    
    if chart_type in ['scatter']:
        kwargs['opacity'] = st.slider("Point Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
        kwargs['size_max'] = st.slider("Max Point Size", min_value=5, max_value=50, value=15, step=5)
    
    if chart_type in ['bar', 'line']:
        kwargs['orientation'] = st.radio("Orientation", options=['v', 'h'], index=0)
    
    # Generate and display the chart
    fig = create_chart(df, chart_type, x_col, y_col, color_col, title, **kwargs)
    st.plotly_chart(fig, use_container_width=True)
    
    return fig
