import streamlit as st
import pandas as pd
import numpy as np
from utils.dashboard_components import (
    get_component_id,
    create_chart_component,
    create_table_component,
    create_filter_component,
    create_metric_component,
    create_text_component
)

def get_available_templates():
    """
    Get a list of available dashboard templates
    
    Returns:
        list: List of template dictionaries with id, name, and description
    """
    return [
        {
            "id": "blank",
            "name": "Blank Dashboard",
            "description": "Start with an empty dashboard",
            "icon": "🔲"
        },
        {
            "id": "basic_analytics",
            "name": "Basic Analytics",
            "description": "Standard analytics with key metrics, charts, and a data table",
            "icon": "📊"
        },
        {
            "id": "executive_summary",
            "name": "Executive Summary",
            "description": "High-level overview with key metrics and trends",
            "icon": "👔"
        },
        {
            "id": "sales_dashboard",
            "name": "Sales Dashboard",
            "description": "Sales performance tracking with metrics and charts",
            "icon": "💰"
        },
        {
            "id": "time_series",
            "name": "Time Series Analysis",
            "description": "Track changes over time with trend analysis",
            "icon": "📈"
        },
        {
            "id": "comparison_dashboard",
            "name": "Comparison Dashboard",
            "description": "Compare different categories or time periods",
            "icon": "⚖️"
        }
    ]

def create_template_dashboard(df, template_id):
    """
    Create a dashboard from a template
    
    Args:
        df: pandas DataFrame
        template_id: Template identifier
        
    Returns:
        list: List of dashboard components
    """
    # Check for required columns based on template
    required_cols = get_template_required_columns(template_id)
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Template requires these column types: {', '.join(required_cols)}")
        st.info("The template will be adapted for your data structure")
    
    # Call the appropriate template function
    if template_id == "blank":
        return create_blank_template()
    elif template_id == "basic_analytics":
        return create_basic_analytics_template(df)
    elif template_id == "executive_summary":
        return create_executive_summary_template(df)
    elif template_id == "sales_dashboard":
        return create_sales_dashboard_template(df)
    elif template_id == "time_series":
        return create_time_series_template(df)
    elif template_id == "comparison_dashboard":
        return create_comparison_template(df)
    else:
        return create_blank_template()

def get_template_required_columns(template_id):
    """
    Get required column types for a template
    
    Args:
        template_id: Template identifier
        
    Returns:
        list: List of required column type descriptions
    """
    if template_id == "blank":
        return []
    elif template_id == "basic_analytics":
        return ["numeric", "categorical"]
    elif template_id == "executive_summary":
        return ["numeric", "categorical", "date/time"]
    elif template_id == "sales_dashboard":
        return ["sales", "product", "date/time", "customer/region"]
    elif template_id == "time_series":
        return ["date/time", "numeric"]
    elif template_id == "comparison_dashboard":
        return ["categorical", "numeric"]
    else:
        return []

def create_blank_template():
    """
    Create a blank dashboard template
    
    Returns:
        list: Empty list of dashboard components
    """
    return []

def create_basic_analytics_template(df):
    """
    Create a basic analytics dashboard template
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: List of dashboard components
    """
    components = []
    
    # Add filter component
    components.append(create_filter_component(df))
    
    # Add title text
    title_text = """
    # Data Analytics Dashboard
    This dashboard provides a basic analysis of the dataset with key metrics and visualizations.
    """
    components.append(create_text_component(content=title_text, title="Dashboard Title"))
    
    # Find relevant columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Add metrics
    if numeric_cols:
        # Create up to 3 metric components
        for i, col in enumerate(numeric_cols[:3]):
            agg_type = ["mean", "sum", "count"][min(i, 2)]
            agg_label = {"mean": "Average", "sum": "Total", "count": "Count"}[agg_type]
            
            metric_settings = {
                "column": col,
                "aggregation": agg_type,
                "title": f"{agg_label} {col}",
                "format": "{:.2f}" if agg_type != "count" else "{:,.0f}"
            }
            components.append(create_metric_component(settings=metric_settings))
    
    # Add charts
    if categorical_cols and numeric_cols:
        # Bar chart showing numeric value by category
        bar_settings = {
            "x_column": categorical_cols[0],
            "y_column": numeric_cols[0],
            "title": f"{numeric_cols[0]} by {categorical_cols[0]}"
        }
        components.append(create_chart_component(chart_type="bar", settings=bar_settings))
    
    if len(numeric_cols) >= 2:
        # Scatter plot showing relationship between two numeric columns
        scatter_settings = {
            "x_column": numeric_cols[0],
            "y_column": numeric_cols[1],
            "title": f"{numeric_cols[1]} vs {numeric_cols[0]}"
        }
        components.append(create_chart_component(chart_type="scatter", settings=scatter_settings))
    
    if numeric_cols:
        # Distribution chart
        hist_settings = {
            "x_column": numeric_cols[0],
            "title": f"Distribution of {numeric_cols[0]}"
        }
        components.append(create_chart_component(chart_type="histogram", settings=hist_settings))
    
    # Add data table
    table_settings = {
        "columns": df.columns.tolist()[:10],  # First 10 columns
        "page_size": 10
    }
    components.append(create_table_component(settings=table_settings))
    
    return components

def create_executive_summary_template(df):
    """
    Create an executive summary dashboard template
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: List of dashboard components
    """
    components = []
    
    # Add title text
    title_text = """
    # Executive Summary Dashboard
    This dashboard provides a high-level overview of key metrics and performance indicators.
    """
    components.append(create_text_component(content=title_text, title="Dashboard Title"))
    
    # Find relevant columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Add metrics section - up to 4 key metrics
    if numeric_cols:
        metrics_title = "## Key Metrics"
        components.append(create_text_component(content=metrics_title, title="Metrics Section"))
        
        # Create up to 4 metric components
        agg_types = ["sum", "mean", "max", "count"]
        agg_labels = {"sum": "Total", "mean": "Average", "max": "Maximum", "count": "Count"}
        
        for i, col in enumerate(numeric_cols[:4]):
            agg_type = agg_types[min(i, 3)]
            agg_label = agg_labels[agg_type]
            
            metric_settings = {
                "column": col,
                "aggregation": agg_type,
                "title": f"{agg_label} {col}",
                "format": "{:,.2f}" if agg_type != "count" else "{:,.0f}"
            }
            components.append(create_metric_component(settings=metric_settings))
    
    # Add trend analysis section
    if date_cols and numeric_cols:
        trends_title = "## Trends Over Time"
        components.append(create_text_component(content=trends_title, title="Trends Section"))
        
        # Line chart showing trend over time
        line_settings = {
            "x_column": date_cols[0],
            "y_column": numeric_cols[0],
            "title": f"{numeric_cols[0]} Trend"
        }
        components.append(create_chart_component(chart_type="line", settings=line_settings))
    
    # Add breakdown section
    if categorical_cols and numeric_cols:
        breakdown_title = "## Breakdown by Category"
        components.append(create_text_component(content=breakdown_title, title="Breakdown Section"))
        
        # Pie chart showing proportion by category
        pie_settings = {
            "x_column": categorical_cols[0],
            "title": f"Distribution by {categorical_cols[0]}"
        }
        components.append(create_chart_component(chart_type="pie", settings=pie_settings))
        
        # Bar chart showing value by category
        bar_settings = {
            "x_column": categorical_cols[0],
            "y_column": numeric_cols[0],
            "title": f"{numeric_cols[0]} by {categorical_cols[0]}"
        }
        components.append(create_chart_component(chart_type="bar", settings=bar_settings))
    
    # Add summary table
    summary_title = "## Summary Table"
    components.append(create_text_component(content=summary_title, title="Summary Section"))
    
    table_settings = {
        "columns": df.columns.tolist()[:6],  # First 6 columns for readability
        "page_size": 5,
        "title": "Summary Data"
    }
    components.append(create_table_component(settings=table_settings))
    
    return components

def create_sales_dashboard_template(df):
    """
    Create a sales dashboard template
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: List of dashboard components
    """
    components = []
    
    # Add filter component
    components.append(create_filter_component(df))
    
    # Add title text
    title_text = """
    # Sales Performance Dashboard
    Track key sales metrics, product performance, and regional analysis.
    """
    components.append(create_text_component(content=title_text, title="Dashboard Title"))
    
    # Try to identify relevant columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Identify potential sales/revenue columns
    sales_cols = [col for col in numeric_cols if any(term in col.lower() for term in ["sale", "revenue", "price", "total", "amount"])]
    
    # Identify potential product columns
    product_cols = [col for col in categorical_cols if any(term in col.lower() for term in ["product", "item", "category", "service"])]
    
    # Identify potential region/customer columns
    region_cols = [col for col in categorical_cols if any(term in col.lower() for term in ["region", "country", "customer", "client", "location"])]
    
    # Add metrics section
    metrics_title = "## Sales Metrics"
    components.append(create_text_component(content=metrics_title, title="Metrics Section"))
    
    # Sales metrics (use identified sales columns or fall back to numeric columns)
    metric_cols = sales_cols if sales_cols else numeric_cols
    
    if metric_cols:
        # Total Sales/Revenue
        total_metric_settings = {
            "column": metric_cols[0],
            "aggregation": "sum",
            "title": f"Total {metric_cols[0]}",
            "format": "${:,.2f}" if "price" in metric_cols[0].lower() or "sale" in metric_cols[0].lower() else "{:,.2f}"
        }
        components.append(create_metric_component(settings=total_metric_settings))
        
        # Average Sales/Value
        if len(metric_cols) > 1:
            avg_col = metric_cols[1]
        else:
            avg_col = metric_cols[0]
            
        avg_metric_settings = {
            "column": avg_col,
            "aggregation": "mean",
            "title": f"Average {avg_col}",
            "format": "${:,.2f}" if "price" in avg_col.lower() or "sale" in avg_col.lower() else "{:,.2f}"
        }
        components.append(create_metric_component(settings=avg_metric_settings))
        
        # Count of Sales/Orders
        count_metric_settings = {
            "column": metric_cols[0],
            "aggregation": "count",
            "title": "Number of Transactions",
            "format": "{:,.0f}"
        }
        components.append(create_metric_component(settings=count_metric_settings))
    
    # Add sales trend section
    if date_cols and metric_cols:
        trends_title = "## Sales Trends"
        components.append(create_text_component(content=trends_title, title="Trends Section"))
        
        # Line chart showing sales over time
        line_settings = {
            "x_column": date_cols[0],
            "y_column": metric_cols[0],
            "title": f"{metric_cols[0]} Over Time"
        }
        components.append(create_chart_component(chart_type="line", settings=line_settings))
    
    # Add product analysis section
    if product_cols and metric_cols:
        product_title = "## Product Performance"
        components.append(create_text_component(content=product_title, title="Product Section"))
        
        # Bar chart showing sales by product
        bar_settings = {
            "x_column": product_cols[0],
            "y_column": metric_cols[0],
            "title": f"{metric_cols[0]} by {product_cols[0]}"
        }
        components.append(create_chart_component(chart_type="bar", settings=bar_settings))
        
        # Pie chart showing product distribution
        pie_settings = {
            "x_column": product_cols[0],
            "title": f"Sales Distribution by {product_cols[0]}"
        }
        components.append(create_chart_component(chart_type="pie", settings=pie_settings))
    
    # Add regional analysis section
    if region_cols and metric_cols:
        region_title = "## Regional Analysis"
        components.append(create_text_component(content=region_title, title="Regional Section"))
        
        # Bar chart showing sales by region
        region_settings = {
            "x_column": region_cols[0],
            "y_column": metric_cols[0],
            "title": f"{metric_cols[0]} by {region_cols[0]}"
        }
        components.append(create_chart_component(chart_type="bar", settings=region_settings))
    
    # Add top performers table
    if product_cols and metric_cols:
        top_title = "## Top Performers"
        components.append(create_text_component(content=top_title, title="Top Performers Section"))
        
        display_cols = []
        if product_cols:
            display_cols.append(product_cols[0])
        if region_cols:
            display_cols.append(region_cols[0])
        if metric_cols:
            display_cols.append(metric_cols[0])
        
        table_settings = {
            "columns": display_cols,
            "page_size": 10,
            "title": "Top Sales Data"
        }
        components.append(create_table_component(settings=table_settings))
    
    return components

def create_time_series_template(df):
    """
    Create a time series analysis dashboard template
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: List of dashboard components
    """
    components = []
    
    # Add filter component
    components.append(create_filter_component(df))
    
    # Add title text
    title_text = """
    # Time Series Analysis Dashboard
    Track trends, patterns, and changes over time with detailed time-based visualizations.
    """
    components.append(create_text_component(content=title_text, title="Dashboard Title"))
    
    # Find relevant columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # If no datetime columns, look for columns that might be dates
    if not date_cols:
        potential_date_cols = [col for col in df.columns if any(date_term in col.lower() for date_term in ["date", "time", "day", "month", "year"])]
        date_cols = potential_date_cols
    
    # Check if we have time data and metrics to plot
    if not date_cols or not numeric_cols:
        no_data_text = """
        ## Missing Required Data
        This template requires time/date columns and numeric columns.
        Please convert date columns to datetime type for best results.
        """
        components.append(create_text_component(content=no_data_text, title="Missing Data Warning"))
        
        # Add a basic table instead
        table_settings = {
            "columns": df.columns.tolist()[:10],
            "page_size": 10
        }
        components.append(create_table_component(settings=table_settings))
        
        return components
    
    # Add metrics for time range
    if date_cols and numeric_cols:
        # Count of data points
        count_metric_settings = {
            "column": numeric_cols[0],
            "aggregation": "count",
            "title": "Number of Data Points",
            "format": "{:,.0f}"
        }
        components.append(create_metric_component(settings=count_metric_settings))
        
        # Average value
        avg_metric_settings = {
            "column": numeric_cols[0],
            "aggregation": "mean",
            "title": f"Average {numeric_cols[0]}",
            "format": "{:,.2f}"
        }
        components.append(create_metric_component(settings=avg_metric_settings))
        
        # Max value
        max_metric_settings = {
            "column": numeric_cols[0],
            "aggregation": "max",
            "title": f"Maximum {numeric_cols[0]}",
            "format": "{:,.2f}"
        }
        components.append(create_metric_component(settings=max_metric_settings))
    
    # Add main trend section
    if date_cols and numeric_cols:
        trend_title = "## Trend Analysis"
        components.append(create_text_component(content=trend_title, title="Trend Section"))
        
        # Line chart showing primary metric over time
        line_settings = {
            "x_column": date_cols[0],
            "y_column": numeric_cols[0],
            "title": f"{numeric_cols[0]} Over Time"
        }
        components.append(create_chart_component(chart_type="line", settings=line_settings))
        
        # Add another line chart if we have multiple metrics
        if len(numeric_cols) > 1:
            line_settings2 = {
                "x_column": date_cols[0],
                "y_column": numeric_cols[1],
                "title": f"{numeric_cols[1]} Over Time"
            }
            components.append(create_chart_component(chart_type="line", settings=line_settings2))
    
    # Add distribution over time section
    if date_cols and numeric_cols:
        dist_title = "## Distribution Analysis"
        components.append(create_text_component(content=dist_title, title="Distribution Section"))
        
        # Box plot showing distribution over time
        box_settings = {
            "x_column": date_cols[0],
            "y_column": numeric_cols[0],
            "title": f"Distribution of {numeric_cols[0]} Over Time"
        }
        components.append(create_chart_component(chart_type="box", settings=box_settings))
    
    # Add cumulative section
    if date_cols and numeric_cols:
        cum_title = "## Cumulative Analysis"
        components.append(create_text_component(content=cum_title, title="Cumulative Section"))
        
        # Area chart showing cumulative values
        area_settings = {
            "x_column": date_cols[0],
            "y_column": numeric_cols[0],
            "title": f"Cumulative {numeric_cols[0]} Over Time"
        }
        components.append(create_chart_component(chart_type="area", settings=area_settings))
    
    # Add data table
    table_title = "## Time Series Data"
    components.append(create_text_component(content=table_title, title="Data Section"))
    
    display_cols = []
    if date_cols:
        display_cols.append(date_cols[0])
    display_cols.extend(numeric_cols[:3])  # Add up to 3 numeric columns
    
    table_settings = {
        "columns": display_cols,
        "page_size": 10,
        "title": "Time Series Data"
    }
    components.append(create_table_component(settings=table_settings))
    
    return components

def create_comparison_template(df):
    """
    Create a comparison dashboard template
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: List of dashboard components
    """
    components = []
    
    # Add filter component
    components.append(create_filter_component(df))
    
    # Add title text
    title_text = """
    # Comparison Dashboard
    Compare metrics across different categories, regions, or time periods.
    """
    components.append(create_text_component(content=title_text, title="Dashboard Title"))
    
    # Find relevant columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Check if we have categorical data and metrics to compare
    if not categorical_cols or not numeric_cols:
        no_data_text = """
        ## Missing Required Data
        This template requires categorical columns and numeric columns for comparison.
        """
        components.append(create_text_component(content=no_data_text, title="Missing Data Warning"))
        
        # Add a basic table instead
        table_settings = {
            "columns": df.columns.tolist()[:10],
            "page_size": 10
        }
        components.append(create_table_component(settings=table_settings))
        
        return components
    
    # Add comparison section
    comp_title = "## Category Comparison"
    components.append(create_text_component(content=comp_title, title="Comparison Section"))
    
    # Bar chart comparing primary metric across categories
    bar_settings = {
        "x_column": categorical_cols[0],
        "y_column": numeric_cols[0],
        "title": f"{numeric_cols[0]} by {categorical_cols[0]}"
    }
    components.append(create_chart_component(chart_type="bar", settings=bar_settings))
    
    # Add distribution comparison
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        dist_title = "## Distribution Comparison"
        components.append(create_text_component(content=dist_title, title="Distribution Section"))
        
        # Box plot comparing distributions
        box_settings = {
            "x_column": categorical_cols[0],
            "y_column": numeric_cols[0],
            "title": f"Distribution of {numeric_cols[0]} by {categorical_cols[0]}"
        }
        components.append(create_chart_component(chart_type="box", settings=box_settings))
    
    # Add proportion comparison
    if len(categorical_cols) >= 2:
        prop_title = "## Proportion Analysis"
        components.append(create_text_component(content=prop_title, title="Proportion Section"))
        
        # Pie chart showing proportions
        pie_settings = {
            "x_column": categorical_cols[1],
            "title": f"Distribution by {categorical_cols[1]}"
        }
        components.append(create_chart_component(chart_type="pie", settings=pie_settings))
    
    # Add correlation section if we have multiple numeric columns
    if len(numeric_cols) >= 2:
        corr_title = "## Correlation Analysis"
        components.append(create_text_component(content=corr_title, title="Correlation Section"))
        
        # Scatter plot showing correlation
        scatter_settings = {
            "x_column": numeric_cols[0],
            "y_column": numeric_cols[1],
            "color_column": categorical_cols[0] if categorical_cols else None,
            "title": f"{numeric_cols[1]} vs {numeric_cols[0]}"
        }
        components.append(create_chart_component(chart_type="scatter", settings=scatter_settings))
    
    # Add data table
    table_title = "## Comparison Data"
    components.append(create_text_component(content=table_title, title="Data Section"))
    
    display_cols = []
    display_cols.extend(categorical_cols[:2])  # Add up to 2 categorical columns
    display_cols.extend(numeric_cols[:2])      # Add up to 2 numeric columns
    
    table_settings = {
        "columns": display_cols,
        "page_size": 10,
        "title": "Comparison Data"
    }
    components.append(create_table_component(settings=table_settings))
    
    return components
