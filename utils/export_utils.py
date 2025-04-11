import base64
import pandas as pd
import numpy as np
import io
import streamlit as st
import plotly.graph_objects as go
from utils.visualization import create_chart
from datetime import datetime

def export_dashboard_as_html(df, components, dashboard_title="Data Dashboard"):
    """
    Export the dashboard as an HTML file
    
    Args:
        df: pandas DataFrame
        components: List of dashboard components
        dashboard_title: Title of the dashboard
        
    Returns:
        str: HTML content as a string
    """
    # Create HTML content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{dashboard_title}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .dashboard-title {{ margin-bottom: 20px; }}
            .component {{ 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 15px; 
                margin-bottom: 20px;
                background-color: white;
            }}
            .component-title {{ margin-top: 0; margin-bottom: 15px; }}
            .dashboard-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
            .chart-container {{ height: 400px; }}
            .table-container {{ overflow-x: auto; }}
            .table {{ width: 100%; border-collapse: collapse; }}
            .table th, .table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            .metric {{ text-align: center; padding: 20px; }}
            .metric-value {{ font-size: 2em; font-weight: bold; }}
            .metric-title {{ font-size: 1.2em; color: #555; }}
            .footer {{ margin-top: 30px; text-align: center; color: #777; font-size: 0.9em; }}
            @media (max-width: 768px) {{
                .dashboard-container {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="dashboard-title">{dashboard_title}</h1>
            <div class="dashboard-container">
    """
    
    # Add components
    for component in components:
        component_type = component.get("type", "")
        component_id = component.get("id", "")
        settings = component.get("settings", {})
        title = component.get("title", "Component")
        
        html += f"""
        <div class="component" id="{component_id}">
            <h3 class="component-title">{title}</h3>
        """
        
        if component_type == "chart":
            html += export_chart_component(df, component)
        elif component_type == "table":
            html += export_table_component(df, component)
        elif component_type == "metric":
            html += export_metric_component(df, component)
        elif component_type == "text":
            html += f"""<div class="text-container">{settings.get('content', '')}</div>"""
        
        html += "</div>"
    
    # Add footer and close tags
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""
            </div>
            <div class="footer">
                Generated on {current_date} | AI-Assisted Data Dashboard Builder
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def export_chart_component(df, component):
    """
    Export a chart component as HTML
    
    Args:
        df: pandas DataFrame
        component: Component configuration
        
    Returns:
        str: HTML content for the chart
    """
    chart_type = component.get("chart_type", "bar")
    settings = component.get("settings", {})
    
    x_column = settings.get("x_column")
    y_column = settings.get("y_column")
    color_column = settings.get("color_column")
    title = settings.get("title", f"{chart_type.title()} Chart")
    
    try:
        # Create the plot
        fig = create_chart(
            df, 
            chart_type=chart_type, 
            x_col=x_column, 
            y_col=y_column, 
            color_col=color_column, 
            title=title
        )
        
        # Convert to HTML
        plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
        
        return f"""
        <div class="chart-container">
            {plot_html}
        </div>
        """
    except Exception as e:
        return f"""<div class="alert alert-danger">Error creating chart: {str(e)}</div>"""

def export_table_component(df, component):
    """
    Export a table component as HTML
    
    Args:
        df: pandas DataFrame
        component: Component configuration
        
    Returns:
        str: HTML content for the table
    """
    settings = component.get("settings", {})
    columns = settings.get("columns", [])
    
    # If specific columns are selected, filter the DataFrame
    if columns and all(col in df.columns for col in columns):
        display_df = df[columns]
    else:
        display_df = df
    
    # Convert DataFrame to HTML table
    table_html = display_df.head(100).to_html(classes="table table-striped", index=False)
    
    return f"""
    <div class="table-container">
        {table_html}
        <p class="mt-2 text-muted">Showing up to 100 rows</p>
    </div>
    """

def export_metric_component(df, component):
    """
    Export a metric component as HTML
    
    Args:
        df: pandas DataFrame
        component: Component configuration
        
    Returns:
        str: HTML content for the metric
    """
    settings = component.get("settings", {})
    
    column = settings.get("column")
    aggregation = settings.get("aggregation", "mean")
    format_str = settings.get("format", "{:.2f}")
    
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
        
        return f"""
        <div class="metric">
            <div class="metric-value">{formatted_value}</div>
            <div class="metric-title">{settings.get('title', column)}</div>
        </div>
        """
    except Exception as e:
        return f"""<div class="alert alert-danger">Error calculating metric: {str(e)}</div>"""

def export_dashboard_as_pdf(df, components, dashboard_title="Data Dashboard"):
    """
    Export the dashboard as a PDF file
    NOTE: This is a placeholder function. PDF generation requires additional libraries like weasyprint
    or a similar solution, which would require additional installation steps.
    
    Args:
        df: pandas DataFrame
        components: List of dashboard components
        dashboard_title: Title of the dashboard
        
    Returns:
        str: Information message about PDF export limitations
    """
    return "PDF export is a premium feature. Please export as HTML instead."

def get_table_download_link(df, filename="data.csv", button_text="Download CSV"):
    """
    Generate a download link for a DataFrame
    
    Args:
        df: pandas DataFrame
        filename: Name of the file to download
        button_text: Text to display on the button
        
    Returns:
        None: Displays the download button directly
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
    return st.markdown(href, unsafe_allow_html=True)

def get_html_download_link(html_content, filename="dashboard.html", button_text="Download HTML"):
    """
    Generate a download link for HTML content
    
    Args:
        html_content: HTML content as a string
        filename: Name of the file to download
        button_text: Text to display on the button
        
    Returns:
        None: Displays the download button directly
    """
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{button_text}</a>'
    return st.markdown(href, unsafe_allow_html=True)

def export_to_excel(df, filename="data.xlsx"):
    """
    Export DataFrame to Excel file
    
    Args:
        df: pandas DataFrame
        filename: Name of the Excel file
        
    Returns:
        bytes: Excel file as bytes
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Data']
        
        # Add a header format
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Write the column headers with the defined format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Set column widths
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(i, i, min(max_len, 30))
    
    return output.getvalue()

def get_excel_download_link(df, filename="data.xlsx", button_text="Download Excel"):
    """
    Generate a download link for Excel file
    
    Args:
        df: pandas DataFrame
        filename: Name of the file to download
        button_text: Text to display on the button
        
    Returns:
        None: Displays the download button directly
    """
    excel_data = export_to_excel(df, filename)
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{button_text}</a>'
    return st.markdown(href, unsafe_allow_html=True)

def export_report_as_markdown(df, components, dashboard_title="Data Dashboard"):
    """
    Export the dashboard as a Markdown report
    
    Args:
        df: pandas DataFrame
        components: List of dashboard components
        dashboard_title: Title of the dashboard
        
    Returns:
        str: Markdown content as a string
    """
    # Create Markdown content
    markdown = f"# {dashboard_title}\n\n"
    markdown += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
    # Dataset overview
    markdown += "## Dataset Overview\n\n"
    markdown += f"- **Rows:** {len(df)}\n"
    markdown += f"- **Columns:** {len(df.columns)}\n"
    
    # Column information
    markdown += "\n### Column Information\n\n"
    markdown += "| Column | Type | Non-Null | Unique Values |\n"
    markdown += "|--------|------|----------|---------------|\n"
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        non_null = df[col].count()
        unique = df[col].nunique()
        markdown += f"| {col} | {col_type} | {non_null} | {unique} |\n"
    
    # Add components
    for component in components:
        component_type = component.get("type", "")
        settings = component.get("settings", {})
        title = component.get("title", "Component")
        
        markdown += f"\n## {title}\n\n"
        
        if component_type == "chart":
            chart_type = component.get("chart_type", "")
            x_column = settings.get("x_column")
            y_column = settings.get("y_column")
            
            markdown += f"*Chart Type: {chart_type}*\n\n"
            
            if x_column:
                markdown += f"X-Axis: {x_column}\n\n"
            if y_column:
                markdown += f"Y-Axis: {y_column}\n\n"
            
            markdown += "*[Chart visualization not available in markdown export]*\n\n"
            
        elif component_type == "table":
            columns = settings.get("columns", [])
            
            if columns and all(col in df.columns for col in columns):
                display_df = df[columns]
            else:
                display_df = df
            
            # Add a sample of the data (first 5 rows)
            markdown += "Sample data:\n\n"
            markdown += display_df.head(5).to_markdown(index=False) + "\n\n"
            
        elif component_type == "metric":
            column = settings.get("column")
            aggregation = settings.get("aggregation", "mean")
            
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
                
                markdown += f"**Value: {value}**\n\n"
                
            except Exception as e:
                markdown += f"*Error calculating metric: {str(e)}*\n\n"
            
        elif component_type == "text":
            markdown += settings.get("content", "") + "\n\n"
    
    # Add footer
    markdown += "---\n"
    markdown += "*Generated by AI-Assisted Data Dashboard Builder*"
    
    return markdown

def get_markdown_download_link(markdown_content, filename="report.md", button_text="Download Markdown"):
    """
    Generate a download link for Markdown content
    
    Args:
        markdown_content: Markdown content as a string
        filename: Name of the file to download
        button_text: Text to display on the button
        
    Returns:
        None: Displays the download button directly
    """
    b64 = base64.b64encode(markdown_content.encode()).decode()
    href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">{button_text}</a>'
    return st.markdown(href, unsafe_allow_html=True)
