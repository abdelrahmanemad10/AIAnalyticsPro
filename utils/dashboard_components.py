import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import uuid
from utils.visualization import create_chart

def show_welcome_page():
    """Display the welcome page when no data is loaded yet"""
    st.title("🚀 AI-Assisted Data Dashboard Builder")
    
    st.markdown("""
    ### Welcome to your data analysis and visualization platform!
    
    This tool helps you:
    - **Explore data** with interactive visualizations
    - **Create custom dashboards** with drag-and-drop components
    - **Gain AI-powered insights** from your data
    - **Generate reports** for sharing with stakeholders
    
    ### Getting Started
    1. Upload your data file using the sidebar (CSV, Excel, JSON, TSV)
    2. Alternatively, click "Use Sample Data" to try the app with our demo dataset
    3. Navigate to different sections using the menu in the sidebar
    
    ### Key Features
    - **Dashboard Builder**: Create custom dashboards with various visualizations
    - **Data Explorer**: Explore and understand your dataset
    - **AI Insights**: Get AI-powered analysis and recommendations
    - **Report Generator**: Create and export professional reports
    """)
    
    # Feature highlights in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Visualizations")
        st.markdown("""
        - Bar charts, line charts
        - Scatter plots, bubble charts
        - Pie charts, treemaps
        - Heatmaps, correlation matrices
        - Geographic maps
        """)
    
    with col2:
        st.markdown("#### 🤖 AI Capabilities")
        st.markdown("""
        - Data pattern detection
        - Chart recommendations
        - Natural language queries
        - Outlier identification
        - Correlation analysis
        - Trend identification
        """)
    
    with col3:
        st.markdown("#### 🔍 Analysis Tools")
        st.markdown("""
        - Data filtering
        - Grouping and aggregation
        - Time series analysis
        - Statistical summaries
        - Cross-tabulation
        - Custom calculations
        """)
    
    st.info("👈 To begin, please upload a data file or use sample data from the sidebar.")

def get_component_id():
    """Generate a unique ID for dashboard components"""
    return str(uuid.uuid4())[:8]

def create_filter_component(df, component_id=None):
    """
    Create a filter component for the dashboard
    
    Args:
        df: pandas DataFrame
        component_id: Unique ID for the component (optional)
        
    Returns:
        dict: Component configuration
    """
    if component_id is None:
        component_id = get_component_id()
    
    # Create a dictionary of column types
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_dtype(df[col]):
            column_types[col] = "datetime"
        else:
            column_types[col] = "categorical"
    
    return {
        "id": component_id,
        "type": "filter",
        "title": "Data Filter",
        "column_types": column_types,
        "settings": {
            "columns": list(df.columns),
            "filter_type": "simple"  # simple, advanced
        }
    }

def create_chart_component(component_id=None, chart_type="bar", settings=None):
    """
    Create a chart component for the dashboard
    
    Args:
        component_id: Unique ID for the component (optional)
        chart_type: Type of chart (default: 'bar')
        settings: Chart settings (optional)
        
    Returns:
        dict: Component configuration
    """
    if component_id is None:
        component_id = get_component_id()
    
    if settings is None:
        settings = {
            "x_column": None,
            "y_column": None,
            "color_column": None,
            "title": f"{chart_type.title()} Chart",
            "description": ""
        }
    
    return {
        "id": component_id,
        "type": "chart",
        "chart_type": chart_type,
        "title": settings.get("title", f"{chart_type.title()} Chart"),
        "settings": settings
    }

def create_table_component(component_id=None, settings=None):
    """
    Create a table component for the dashboard
    
    Args:
        component_id: Unique ID for the component (optional)
        settings: Table settings (optional)
        
    Returns:
        dict: Component configuration
    """
    if component_id is None:
        component_id = get_component_id()
    
    if settings is None:
        settings = {
            "columns": [],
            "page_size": 10,
            "sortable": True,
            "title": "Data Table"
        }
    
    return {
        "id": component_id,
        "type": "table",
        "title": settings.get("title", "Data Table"),
        "settings": settings
    }

def create_metric_component(component_id=None, settings=None):
    """
    Create a metric component for the dashboard
    
    Args:
        component_id: Unique ID for the component (optional)
        settings: Metric settings (optional)
        
    Returns:
        dict: Component configuration
    """
    if component_id is None:
        component_id = get_component_id()
    
    if settings is None:
        settings = {
            "column": None,
            "aggregation": "mean",  # mean, sum, min, max, count
            "title": "Metric",
            "format": "{:.2f}",
            "comparison_value": None,
            "comparison_type": None  # percent, absolute
        }
    
    return {
        "id": component_id,
        "type": "metric",
        "title": settings.get("title", "Metric"),
        "settings": settings
    }

def create_text_component(component_id=None, content="", title="Text Block"):
    """
    Create a text component for the dashboard
    
    Args:
        component_id: Unique ID for the component (optional)
        content: Text content (default: '')
        title: Component title (default: 'Text Block')
        
    Returns:
        dict: Component configuration
    """
    if component_id is None:
        component_id = get_component_id()
    
    return {
        "id": component_id,
        "type": "text",
        "title": title,
        "settings": {
            "content": content
        }
    }

def render_component(df, component, filtered_df=None):
    """
    Render a dashboard component
    
    Args:
        df: Original pandas DataFrame
        component: Component configuration dictionary
        filtered_df: Filtered pandas DataFrame (optional)
        
    Returns:
        None
    """
    if filtered_df is None:
        filtered_df = df
    
    component_type = component.get("type", "")
    component_id = component.get("id", "")
    settings = component.get("settings", {})
    
    # Create container with border and padding
    with st.container(border=True):
        # Component header with title and options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(component.get("title", "Component"))
        
        with col2:
            st.caption(f"ID: {component_id}")
        
        # Render different component types
        if component_type == "chart":
            render_chart_component(filtered_df, component)
            
        elif component_type == "table":
            render_table_component(filtered_df, component)
            
        elif component_type == "filter":
            render_filter_component(df, component)
            
        elif component_type == "metric":
            render_metric_component(filtered_df, component)
            
        elif component_type == "text":
            render_text_component(component)
            
        else:
            st.warning(f"Unknown component type: {component_type}")

def render_chart_component(df, component):
    """
    Render a chart component
    
    Args:
        df: pandas DataFrame
        component: Component configuration dictionary
        
    Returns:
        None
    """
    chart_type = component.get("chart_type", "bar")
    settings = component.get("settings", {})
    
    x_column = settings.get("x_column")
    y_column = settings.get("y_column")
    color_column = settings.get("color_column")
    title = settings.get("title", f"{chart_type.title()} Chart")
    
    # Check if necessary columns are selected
    if x_column not in df.columns:
        st.warning(f"X-axis column '{x_column}' not found in data. Please configure the chart.")
        return
    
    if chart_type in ["scatter", "line", "bar"] and y_column not in df.columns:
        st.warning(f"Y-axis column '{y_column}' not found in data. Please configure the chart.")
        return
    
    # Validate color column if specified
    if color_column and color_column not in df.columns:
        st.warning(f"Color column '{color_column}' not found in data.")
        color_column = None
    
    # Create and display the chart
    try:
        fig = create_chart(
            df, 
            chart_type=chart_type, 
            x_col=x_column, 
            y_col=y_column, 
            color_col=color_column, 
            title=title
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

def render_table_component(df, component):
    """
    Render a table component
    
    Args:
        df: pandas DataFrame
        component: Component configuration dictionary
        
    Returns:
        None
    """
    settings = component.get("settings", {})
    
    columns = settings.get("columns", [])
    page_size = settings.get("page_size", 10)
    
    # If specific columns are selected, filter the DataFrame
    if columns and all(col in df.columns for col in columns):
        display_df = df[columns]
    else:
        display_df = df
    
    # Display table with pagination
    if len(display_df) > page_size:
        # Add pagination
        page_number = st.number_input(
            "Page", 
            min_value=1, 
            max_value=max(1, len(display_df) // page_size + 1),
            value=1,
            key=f"page_{component['id']}"
        )
        
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size
        
        st.dataframe(
            display_df.iloc[start_idx:end_idx],
            use_container_width=True
        )
        
        st.caption(f"Showing {start_idx+1}-{min(end_idx, len(display_df))} of {len(display_df)} rows")
    else:
        st.dataframe(display_df, use_container_width=True)

def render_filter_component(df, component):
    """
    Render a filter component
    
    Args:
        df: pandas DataFrame
        component: Component configuration dictionary
        
    Returns:
        dict: Applied filters
    """
    settings = component.get("settings", {})
    column_types = component.get("column_types", {})
    
    # Initialize filters dictionary
    filters = {}
    
    # Allow user to select columns to filter on
    filter_columns = st.multiselect(
        "Select columns to filter",
        options=df.columns.tolist(),
        key=f"filter_cols_{component['id']}"
    )
    
    # Create filters for selected columns
    for col in filter_columns:
        if col in df.columns:
            # Determine column type
            col_type = column_types.get(col, "categorical")
            
            if col_type == "numeric":
                # Numeric filter with min/max slider
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                
                if min_val == max_val:
                    st.text(f"{col}: {min_val}")
                    filters[col] = min_val
                else:
                    filter_range = st.slider(
                        f"Filter by {col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"filter_{component['id']}_{col}"
                    )
                    filters[col] = filter_range
                    
            elif col_type == "datetime":
                # Date filter with date picker
                min_date = df[col].min().date()
                max_date = df[col].max().date()
                
                if min_date == max_date:
                    st.text(f"{col}: {min_date}")
                    filters[col] = min_date
                else:
                    start_date = st.date_input(
                        f"Start date for {col}",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key=f"filter_{component['id']}_{col}_start"
                    )
                    
                    end_date = st.date_input(
                        f"End date for {col}",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date,
                        key=f"filter_{component['id']}_{col}_end"
                    )
                    
                    filters[col] = (pd.Timestamp(start_date), pd.Timestamp(end_date))
                    
            else:
                # Categorical filter with multiselect
                options = df[col].dropna().unique().tolist()
                selected = st.multiselect(
                    f"Filter by {col}",
                    options=options,
                    default=options[:min(5, len(options))],
                    key=f"filter_{component['id']}_{col}"
                )
                
                if selected:
                    filters[col] = selected
    
    return filters

def render_metric_component(df, component):
    """
    Render a metric component
    
    Args:
        df: pandas DataFrame
        component: Component configuration dictionary
        
    Returns:
        None
    """
    settings = component.get("settings", {})
    
    column = settings.get("column")
    aggregation = settings.get("aggregation", "mean")
    format_str = settings.get("format", "{:.2f}")
    comparison_value = settings.get("comparison_value")
    comparison_type = settings.get("comparison_type")
    
    # Check if column exists
    if column not in df.columns:
        st.warning(f"Column '{column}' not found in data. Please configure the metric.")
        return
    
    # Calculate the metric value
    try:
        if aggregation == "mean":
            value = df[column].mean()
        elif aggregation == "sum":
            value = df[column].sum()
        elif aggregation == "min":
            value = df[column].min()
        elif aggregation == "max":
            value = df[column].max()
        elif aggregation == "count":
            value = df[column].count()
        else:
            value = df[column].mean()
        
        # Format the value
        formatted_value = format_str.format(value)
        
        # Calculate delta for comparison if provided
        delta = None
        if comparison_value is not None:
            if comparison_type == "percent":
                delta = f"{((value - comparison_value) / comparison_value) * 100:.1f}%"
            else:
                delta = value - comparison_value
        
        # Display the metric
        st.metric(
            label=settings.get("title", column),
            value=formatted_value,
            delta=delta
        )
        
    except Exception as e:
        st.error(f"Error calculating metric: {str(e)}")

def render_text_component(component):
    """
    Render a text component
    
    Args:
        component: Component configuration dictionary
        
    Returns:
        None
    """
    settings = component.get("settings", {})
    content = settings.get("content", "")
    
    # Display the text content
    st.markdown(content)

def get_default_dashboard(df):
    """
    Create a default dashboard based on the dataset
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: List of dashboard components
    """
    components = []
    
    # Add a filter component
    components.append(create_filter_component(df))
    
    # Try to identify datetime columns for trends
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to identify numeric columns for metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Try to identify categorical columns for grouping
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Add components based on data types
    
    # Add metrics if we have numeric columns
    if numeric_cols:
        metric_settings = {
            "column": numeric_cols[0],
            "aggregation": "mean",
            "title": f"Average {numeric_cols[0]}",
            "format": "{:.2f}"
        }
        components.append(create_metric_component(settings=metric_settings))
        
        if len(numeric_cols) > 1:
            metric_settings2 = {
                "column": numeric_cols[1],
                "aggregation": "sum",
                "title": f"Total {numeric_cols[1]}",
                "format": "{:.2f}"
            }
            components.append(create_metric_component(settings=metric_settings2))
    
    # Add a time series chart if we have date and numeric columns
    if date_cols and numeric_cols:
        chart_settings = {
            "x_column": date_cols[0],
            "y_column": numeric_cols[0],
            "title": f"{numeric_cols[0]} over Time"
        }
        components.append(create_chart_component(chart_type="line", settings=chart_settings))
    
    # Add a bar chart for categorical data
    if categorical_cols and numeric_cols:
        chart_settings = {
            "x_column": categorical_cols[0],
            "y_column": numeric_cols[0],
            "title": f"{numeric_cols[0]} by {categorical_cols[0]}"
        }
        components.append(create_chart_component(chart_type="bar", settings=chart_settings))
    
    # Add a table component with all columns
    table_settings = {
        "columns": df.columns.tolist()[:10],  # First 10 columns for readability
        "page_size": 10
    }
    components.append(create_table_component(settings=table_settings))
    
    return components

def save_dashboard_to_session(dashboard_components, dashboard_name="My Dashboard"):
    """
    Save the current dashboard to session state
    
    Args:
        dashboard_components: List of dashboard components
        dashboard_name: Name of the dashboard (default: 'My Dashboard')
        
    Returns:
        None
    """
    # Initialize saved_dashboards if it doesn't exist
    if 'saved_dashboards' not in st.session_state:
        st.session_state.saved_dashboards = {}
    
    # Save the dashboard
    st.session_state.saved_dashboards[dashboard_name] = dashboard_components.copy()

def load_dashboard_from_session(dashboard_name):
    """
    Load a dashboard from session state
    
    Args:
        dashboard_name: Name of the dashboard to load
        
    Returns:
        list: List of dashboard components, or None if not found
    """
    if 'saved_dashboards' in st.session_state and dashboard_name in st.session_state.saved_dashboards:
        return st.session_state.saved_dashboards[dashboard_name].copy()
    return None
