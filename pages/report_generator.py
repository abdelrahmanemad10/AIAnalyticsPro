import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io
from utils.export_utils import (
    export_dashboard_as_html,
    export_dashboard_as_pdf,
    get_table_download_link,
    get_html_download_link,
    get_excel_download_link,
    export_report_as_markdown,
    get_markdown_download_link
)
from utils.dashboard_components import (
    render_component,
    create_text_component,
    create_metric_component,
    create_chart_component,
    create_table_component
)

st.set_page_config(
    page_title="Report Generator | AI-Assisted Data Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("Please load a dataset first!")
    st.page_link("app.py", label="Go to Home Page", icon="🏠")
    st.stop()

# Get the data
df = st.session_state.data

# Initialize report states
if 'report_components' not in st.session_state:
    st.session_state.report_components = []
if 'report_title' not in st.session_state:
    st.session_state.report_title = "Data Analysis Report"
if 'report_description' not in st.session_state:
    st.session_state.report_description = f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d')}"

# Function to add a component to the report
def add_report_component(component_type, chart_type=None):
    from utils.dashboard_components import (
        get_component_id,
        create_chart_component,
        create_table_component,
        create_metric_component,
        create_text_component
    )
    
    if component_type == "chart":
        component = create_chart_component(chart_type=chart_type)
    elif component_type == "table":
        component = create_table_component()
    elif component_type == "metric":
        component = create_metric_component()
    elif component_type == "text":
        component = create_text_component()
    else:
        return
    
    st.session_state.report_components.append(component)

# Function to remove a component
def remove_report_component(index):
    if 0 <= index < len(st.session_state.report_components):
        st.session_state.report_components.pop(index)

# Function to add dashboard components to report
def add_dashboard_to_report():
    if 'dashboard_components' in st.session_state and st.session_state.dashboard_components:
        # Add all dashboard components to the report
        for component in st.session_state.dashboard_components:
            st.session_state.report_components.append(component.copy())
        return True
    return False

# Sidebar for report controls
with st.sidebar:
    st.title("Report Generator")
    
    # Report metadata
    st.subheader("Report Information")
    st.session_state.report_title = st.text_input("Report Title", value=st.session_state.report_title)
    st.session_state.report_description = st.text_area("Description", value=st.session_state.report_description, height=100)
    
    # Dashboard import
    st.subheader("Import From Dashboard")
    if st.button("Import Dashboard Components", use_container_width=True):
        if add_dashboard_to_report():
            st.success("Dashboard components added to report!")
        else:
            st.warning("No dashboard components found to import.")
    
    # Component controls
    st.subheader("Add Components")
    
    component_col1, component_col2 = st.columns(2)
    
    with component_col1:
        if st.button("Add Chart", key="report_add_chart", use_container_width=True):
            add_report_component("chart", "bar")
            st.rerun()
        
        if st.button("Add Table", key="report_add_table", use_container_width=True):
            add_report_component("table")
            st.rerun()
    
    with component_col2:
        if st.button("Add Metric", key="report_add_metric", use_container_width=True):
            add_report_component("metric")
            st.rerun()
        
        if st.button("Add Text", key="report_add_text", use_container_width=True):
            add_report_component("text")
            st.rerun()
    
    # Clear report
    if st.button("Clear Report", use_container_width=True):
        st.session_state.report_components = []
        st.rerun()
    
    # Export options
    st.subheader("Export Options")
    
    export_format = st.selectbox(
        "Export Format",
        options=["HTML", "Markdown", "PDF", "Excel (Data Only)", "CSV (Data Only)"]
    )
    
    if st.button("Generate Export", use_container_width=True):
        if export_format == "HTML":
            html_content = export_dashboard_as_html(
                df, 
                st.session_state.report_components,
                st.session_state.report_title
            )
            get_html_download_link(html_content, "report.html", "📥 Download HTML Report")
        
        elif export_format == "PDF":
            result = export_dashboard_as_pdf(
                df,
                st.session_state.report_components,
                st.session_state.report_title
            )
            st.info(result)
        
        elif export_format == "Markdown":
            markdown_content = export_report_as_markdown(
                df,
                st.session_state.report_components,
                st.session_state.report_title
            )
            get_markdown_download_link(markdown_content, "report.md", "📥 Download Markdown Report")
        
        elif export_format == "Excel (Data Only)":
            get_excel_download_link(df, "data.xlsx", "📥 Download Excel Data")
        
        elif export_format == "CSV (Data Only)":
            get_table_download_link(df, "data.csv", "📥 Download CSV Data")

# Main content
st.title("📄 Report Generator")
st.write("Create and export professional reports from your data analysis.")

# Report building interface
st.header(st.session_state.report_title)
st.write(st.session_state.report_description)

# Check if report is empty
if not st.session_state.report_components:
    st.info("""
    Your report is empty. You can add components using the sidebar or import from your dashboard.
    
    Tips for creating effective reports:
    - Start with a clear title and description that summarizes the purpose
    - Include key metrics and visualizations that tell a story
    - Add text sections to explain your findings and insights
    - Organize content in a logical flow from overview to details
    """)
else:
    # Report preview and component editor
    tab1, tab2 = st.tabs(["Report Preview", "Edit Components"])
    
    with tab1:
        # Show the report with live components
        for component in st.session_state.report_components:
            render_component(df, component)
    
    with tab2:
        # Component editor
        st.subheader("Edit Report Components")
        
        for i, component in enumerate(st.session_state.report_components):
            with st.expander(f"{i+1}. {component.get('title', 'Component')} ({component.get('type', 'unknown').title()})"):
                # Component settings
                component_type = component.get("type", "")
                settings = component.get("settings", {})
                
                # Common settings for all components
                component["title"] = st.text_input("Component Title", value=component.get("title", ""), key=f"report_title_{i}")
                
                # Type-specific settings
                if component_type == "chart":
                    chart_type = component.get("chart_type", "bar")
                    chart_options = ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "area"]
                    
                    selected_chart = st.selectbox(
                        "Chart Type",
                        options=chart_options,
                        index=chart_options.index(chart_type) if chart_type in chart_options else 0,
                        key=f"report_chart_type_{i}"
                    )
                    component["chart_type"] = selected_chart
                    
                    # Column selectors for chart
                    x_col = settings.get("x_column")
                    x_options = [None] + df.columns.tolist()
                    x_index = x_options.index(x_col) if x_col in x_options else 0
                    
                    settings["x_column"] = st.selectbox(
                        "X-Axis Column",
                        options=x_options,
                        index=x_index,
                        key=f"report_x_col_{i}"
                    )
                    
                    # Y-axis for charts that need it
                    if selected_chart in ["bar", "line", "scatter", "box"]:
                        y_col = settings.get("y_column")
                        y_options = [None] + df.columns.tolist()
                        y_index = y_options.index(y_col) if y_col in y_options else 0
                        
                        settings["y_column"] = st.selectbox(
                            "Y-Axis Column",
                            options=y_options,
                            index=y_index,
                            key=f"report_y_col_{i}"
                        )
                    
                    # Color column
                    color_col = settings.get("color_column")
                    color_options = [None] + df.columns.tolist()
                    color_index = color_options.index(color_col) if color_col in color_options else 0
                    
                    settings["color_column"] = st.selectbox(
                        "Color By",
                        options=color_options,
                        index=color_index,
                        key=f"report_color_col_{i}"
                    )
                
                elif component_type == "table":
                    # Column selector for table
                    all_columns = df.columns.tolist()
                    selected_columns = settings.get("columns", all_columns)
                    
                    settings["columns"] = st.multiselect(
                        "Columns to Display",
                        options=all_columns,
                        default=selected_columns,
                        key=f"report_table_cols_{i}"
                    )
                    
                    # Page size
                    settings["page_size"] = st.number_input(
                        "Rows per page",
                        min_value=5,
                        max_value=100,
                        value=settings.get("page_size", 10),
                        step=5,
                        key=f"report_page_size_{i}"
                    )
                
                elif component_type == "metric":
                    # Column selector for metric
                    metric_col = settings.get("column")
                    metric_options = [None] + df.select_dtypes(include=[np.number]).columns.tolist()
                    metric_index = metric_options.index(metric_col) if metric_col in metric_options else 0
                    
                    settings["column"] = st.selectbox(
                        "Metric Column",
                        options=metric_options,
                        index=metric_index,
                        key=f"report_metric_col_{i}"
                    )
                    
                    # Aggregation method
                    agg_options = ["mean", "sum", "min", "max", "count"]
                    agg_index = agg_options.index(settings.get("aggregation", "mean"))
                    
                    settings["aggregation"] = st.selectbox(
                        "Aggregation Method",
                        options=agg_options,
                        index=agg_index,
                        key=f"report_agg_method_{i}"
                    )
                    
                    # Format string
                    settings["format"] = st.text_input(
                        "Format String (e.g., ${:.2f})",
                        value=settings.get("format", "{:.2f}"),
                        key=f"report_format_{i}"
                    )
                
                elif component_type == "text":
                    # Text content
                    settings["content"] = st.text_area(
                        "Text Content",
                        value=settings.get("content", ""),
                        height=150,
                        key=f"report_text_content_{i}"
                    )
                
                # Component actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Remove", key=f"report_remove_{i}", use_container_width=True):
                        remove_report_component(i)
                        st.rerun()
                
                with col2:
                    if i > 0 and st.button("Move Up", key=f"report_up_{i}", use_container_width=True):
                        # Swap with previous component
                        st.session_state.report_components[i], st.session_state.report_components[i-1] = \
                        st.session_state.report_components[i-1], st.session_state.report_components[i]
                        st.rerun()
                
                with col3:
                    if i < len(st.session_state.report_components) - 1 and st.button("Move Down", key=f"report_down_{i}", use_container_width=True):
                        # Swap with next component
                        st.session_state.report_components[i], st.session_state.report_components[i+1] = \
                        st.session_state.report_components[i+1], st.session_state.report_components[i]
                        st.rerun()

# Report templates section
if not st.session_state.report_components:
    st.header("Report Templates")
    st.write("Choose a template to get started quickly:")
    
    template_col1, template_col2, template_col3 = st.columns(3)
    
    with template_col1:
        with st.container(border=True):
            st.subheader("Executive Summary")
            st.write("A high-level overview with key metrics and insights.")
            if st.button("Use Template", key="template_exec", use_container_width=True):
                # Create executive summary template
                st.session_state.report_title = "Executive Summary Report"
                st.session_state.report_description = f"A high-level overview of the data analysis. Generated on {datetime.datetime.now().strftime('%Y-%m-%d')}."
                
                # Add header text
                intro_text = """
                # Executive Summary
                
                This report provides a high-level overview of the key findings from the data analysis.
                The following sections highlight the most important metrics and trends observed.
                """
                st.session_state.report_components.append(create_text_component(content=intro_text, title="Introduction"))
                
                # Add metrics if we have numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    for i, col in enumerate(numeric_cols[:3]):  # Add up to 3 metrics
                        agg_type = ["mean", "sum", "max"][i % 3]
                        agg_label = {"mean": "Average", "sum": "Total", "max": "Maximum"}[agg_type]
                        settings = {
                            "column": col,
                            "aggregation": agg_type,
                            "title": f"{agg_label} {col}",
                            "format": "{:.2f}"
                        }
                        st.session_state.report_components.append(create_metric_component(settings=settings))
                
                # Add key findings text
                findings_text = """
                ## Key Findings
                
                * Finding 1: Replace with your first key finding
                * Finding 2: Replace with your second key finding
                * Finding 3: Replace with your third key finding
                
                ## Recommendations
                
                Based on the analysis, we recommend the following actions:
                
                1. First recommendation
                2. Second recommendation
                3. Third recommendation
                """
                st.session_state.report_components.append(create_text_component(content=findings_text, title="Key Findings and Recommendations"))
                
                # Add a chart
                if numeric_cols and len(df.columns) > 1:
                    # Find a categorical column for x-axis if available
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    x_col = cat_cols[0] if cat_cols else df.columns[0]
                    y_col = numeric_cols[0]
                    
                    chart_settings = {
                        "x_column": x_col,
                        "y_column": y_col,
                        "title": f"{y_col} by {x_col}"
                    }
                    st.session_state.report_components.append(create_chart_component(chart_type="bar", settings=chart_settings))
                
                st.rerun()
    
    with template_col2:
        with st.container(border=True):
            st.subheader("Data Analysis Report")
            st.write("A comprehensive analysis with visualizations and insights.")
            if st.button("Use Template", key="template_analysis", use_container_width=True):
                # Create data analysis report template
                st.session_state.report_title = "Data Analysis Report"
                st.session_state.report_description = f"A comprehensive analysis of the dataset. Generated on {datetime.datetime.now().strftime('%Y-%m-%d')}."
                
                # Add introduction
                intro_text = """
                # Data Analysis Report
                
                ## Introduction
                
                This report presents a comprehensive analysis of the dataset. The analysis 
                includes data overview, exploratory visualizations, and key insights derived 
                from the data.
                
                ## Data Overview
                
                The dataset contains information about [describe your dataset here]. 
                It includes [number of rows] records with [number of columns] variables.
                """
                st.session_state.report_components.append(create_text_component(content=intro_text, title="Introduction"))
                
                # Add data table
                table_settings = {
                    "columns": df.columns.tolist()[:8],  # First 8 columns for readability
                    "page_size": 5,
                    "title": "Data Sample"
                }
                st.session_state.report_components.append(create_table_component(settings=table_settings))
                
                # Add exploratory analysis section
                eda_text = """
                ## Exploratory Data Analysis
                
                The following visualizations highlight key patterns and relationships in the data.
                """
                st.session_state.report_components.append(create_text_component(content=eda_text, title="Exploratory Analysis"))
                
                # Add visualizations
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if numeric_cols:
                    # Distribution chart
                    hist_settings = {
                        "x_column": numeric_cols[0],
                        "title": f"Distribution of {numeric_cols[0]}"
                    }
                    st.session_state.report_components.append(create_chart_component(chart_type="histogram", settings=hist_settings))
                
                if categorical_cols and numeric_cols:
                    # Bar chart
                    bar_settings = {
                        "x_column": categorical_cols[0],
                        "y_column": numeric_cols[0],
                        "title": f"{numeric_cols[0]} by {categorical_cols[0]}"
                    }
                    st.session_state.report_components.append(create_chart_component(chart_type="bar", settings=bar_settings))
                
                # Add insights and conclusion
                conclusion_text = """
                ## Key Insights
                
                Based on the analysis, we can derive the following insights:
                
                1. First insight
                2. Second insight
                3. Third insight
                
                ## Conclusion
                
                [Your conclusion here]
                
                ## Next Steps
                
                [Recommended next steps based on the analysis]
                """
                st.session_state.report_components.append(create_text_component(content=conclusion_text, title="Conclusion"))
                
                st.rerun()
    
    with template_col3:
        with st.container(border=True):
            st.subheader("Performance Dashboard")
            st.write("A metrics-focused report with key performance indicators.")
            if st.button("Use Template", key="template_performance", use_container_width=True):
                # Create performance dashboard template
                st.session_state.report_title = "Performance Dashboard"
                st.session_state.report_description = f"Key performance metrics and trends. Generated on {datetime.datetime.now().strftime('%Y-%m-%d')}."
                
                # Add header
                header_text = """
                # Performance Dashboard
                
                This dashboard presents key performance indicators and metrics for tracking
                progress toward business objectives.
                """
                st.session_state.report_components.append(create_text_component(content=header_text, title="Dashboard Header"))
                
                # Add KPI metrics
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    kpi_metrics = [
                        {"col": numeric_cols[0], "agg": "sum", "title": "Total", "format": "{:,.2f}"},
                        {"col": numeric_cols[min(1, len(numeric_cols)-1)], "agg": "mean", "title": "Average", "format": "{:,.2f}"},
                        {"col": numeric_cols[min(2, len(numeric_cols)-1)], "agg": "count", "title": "Count", "format": "{:,}"}
                    ]
                    
                    for kpi in kpi_metrics:
                        settings = {
                            "column": kpi["col"],
                            "aggregation": kpi["agg"],
                            "title": f"{kpi['title']} {kpi['col']}",
                            "format": kpi["format"]
                        }
                        st.session_state.report_components.append(create_metric_component(settings=settings))
                
                # Add performance charts
                date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                
                # Try to identify if we have time data
                if not date_cols:
                    potential_date_cols = [col for col in df.columns if any(term in col.lower() for term in ["date", "time", "day", "month", "year"])]
                    date_cols = potential_date_cols
                
                if date_cols and numeric_cols:
                    # Time series chart
                    line_settings = {
                        "x_column": date_cols[0],
                        "y_column": numeric_cols[0],
                        "title": f"{numeric_cols[0]} Over Time"
                    }
                    st.session_state.report_components.append(create_chart_component(chart_type="line", settings=line_settings))
                
                # Add category breakdown
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_cols and numeric_cols:
                    # Bar chart
                    bar_settings = {
                        "x_column": categorical_cols[0],
                        "y_column": numeric_cols[0],
                        "title": f"{numeric_cols[0]} by {categorical_cols[0]}"
                    }
                    st.session_state.report_components.append(create_chart_component(chart_type="bar", settings=bar_settings))
                
                # Add summary text
                summary_text = """
                ## Performance Summary
                
                * Area 1: [Summary of performance in this area]
                * Area 2: [Summary of performance in this area]
                * Area 3: [Summary of performance in this area]
                
                ## Action Items
                
                1. [Action item based on performance data]
                2. [Action item based on performance data]
                3. [Action item based on performance data]
                """
                st.session_state.report_components.append(create_text_component(content=summary_text, title="Summary"))
                
                st.rerun()
