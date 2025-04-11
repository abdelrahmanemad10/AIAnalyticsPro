import streamlit as st
import pandas as pd
import numpy as np
import json
from utils.dashboard_components import (
    get_component_id,
    create_chart_component,
    create_table_component,
    create_filter_component,
    create_metric_component,
    create_text_component,
    render_component,
    get_default_dashboard,
    save_dashboard_to_session
)
from utils.visualization import (
    get_recommended_charts,
    get_bivariate_chart_recommendations,
    create_chart
)
from templates.dashboard_templates import (
    get_available_templates,
    create_template_dashboard
)
from utils.data_processor import filter_dataframe

st.set_page_config(
    page_title="Dashboard Builder | AI-Assisted Data Dashboard",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("Please load a dataset first!")
    st.page_link("app.py", label="Go to Home Page", icon="🏠")
    st.stop()

# Initialize dashboard components if not already in session state
if 'dashboard_components' not in st.session_state:
    st.session_state.dashboard_components = []

if 'dashboard_name' not in st.session_state:
    st.session_state.dashboard_name = "My Dashboard"

if 'applied_filters' not in st.session_state:
    st.session_state.applied_filters = {}

# Function to add a new component
def add_component(component_type, chart_type=None):
    if component_type == "chart":
        component = create_chart_component(chart_type=chart_type)
    elif component_type == "table":
        component = create_table_component()
    elif component_type == "metric":
        component = create_metric_component()
    elif component_type == "text":
        component = create_text_component()
    elif component_type == "filter":
        component = create_filter_component(st.session_state.data)
    else:
        return
    
    st.session_state.dashboard_components.append(component)

# Function to remove a component
def remove_component(index):
    if 0 <= index < len(st.session_state.dashboard_components):
        st.session_state.dashboard_components.pop(index)
        
# Function to move a component up
def move_component_up(index):
    if 0 < index < len(st.session_state.dashboard_components):
        st.session_state.dashboard_components[index], st.session_state.dashboard_components[index-1] = \
        st.session_state.dashboard_components[index-1], st.session_state.dashboard_components[index]

# Function to move a component down
def move_component_down(index):
    if 0 <= index < len(st.session_state.dashboard_components) - 1:
        st.session_state.dashboard_components[index], st.session_state.dashboard_components[index+1] = \
        st.session_state.dashboard_components[index+1], st.session_state.dashboard_components[index]

# Sidebar for dashboard controls
with st.sidebar:
    st.title("Dashboard Builder")
    
    # Dashboard name
    st.session_state.dashboard_name = st.text_input("Dashboard Name", value=st.session_state.dashboard_name)
    
    # Dashboard templates
    st.subheader("Dashboard Templates")
    templates = get_available_templates()
    template_options = {template["name"]: template["id"] for template in templates}
    
    template_col1, template_col2 = st.columns([3, 1])
    with template_col1:
        selected_template = st.selectbox(
            "Select a template",
            options=list(template_options.keys()),
            index=0
        )
    
    with template_col2:
        if st.button("Apply", use_container_width=True):
            template_id = template_options[selected_template]
            st.session_state.dashboard_components = create_template_dashboard(st.session_state.data, template_id)
            st.session_state.current_template = template_id
            st.rerun()
    
    # Component controls
    st.subheader("Add Components")
    
    component_col1, component_col2 = st.columns(2)
    
    with component_col1:
        if st.button("Add Chart", use_container_width=True):
            add_component("chart", "bar")
            st.rerun()
        
        if st.button("Add Table", use_container_width=True):
            add_component("table")
            st.rerun()
    
    with component_col2:
        if st.button("Add Metric", use_container_width=True):
            add_component("metric")
            st.rerun()
        
        if st.button("Add Text", use_container_width=True):
            add_component("text")
            st.rerun()
    
    if st.button("Add Filter", use_container_width=True):
        add_component("filter")
        st.rerun()
    
    st.divider()
    
    # Save dashboard
    st.subheader("Save Dashboard")
    save_name = st.text_input("Save As", value=st.session_state.dashboard_name)
    
    if st.button("Save Dashboard", use_container_width=True):
        save_dashboard_to_session(st.session_state.dashboard_components, save_name)
        st.success(f"Dashboard '{save_name}' saved!")
    
    # Clear dashboard
    if st.button("Clear Dashboard", use_container_width=True):
        st.session_state.dashboard_components = []
        st.rerun()

# Main content
st.title("🔨 Dashboard Builder")
st.write("Create your custom dashboard by adding and configuring components.")

# Check if dashboard is empty
if not st.session_state.dashboard_components:
    st.info("Your dashboard is empty. Add components using the sidebar or choose a template to get started.")
    
    # AI recommendations section
    st.header("AI-Recommended Dashboard")
    st.write("Let us create a dashboard based on your data:")
    
    if st.button("Generate Recommended Dashboard"):
        with st.spinner("Generating dashboard recommendations..."):
            # Create a default dashboard with the most relevant visualizations
            st.session_state.dashboard_components = get_default_dashboard(st.session_state.data)
            st.rerun()
else:
    # Dashboard tabs
    tab1, tab2 = st.tabs(["Dashboard View", "Edit Components"])
    
    with tab1:
        # Show the dashboard with live components
        st.subheader(st.session_state.dashboard_name)
        
        # Apply filters to create a filtered dataframe
        filtered_df = st.session_state.data.copy()
        for component in st.session_state.dashboard_components:
            if component.get("type") == "filter":
                if st.session_state.applied_filters:
                    filtered_df = filter_dataframe(filtered_df, st.session_state.applied_filters)
        
        # Render all components
        for component in st.session_state.dashboard_components:
            render_component(st.session_state.data, component, filtered_df)
    
    with tab2:
        # Component editor
        st.subheader("Edit Dashboard Components")
        
        for i, component in enumerate(st.session_state.dashboard_components):
            with st.expander(f"{i+1}. {component.get('title', 'Component')} ({component.get('type', 'unknown').title()})"):
                # Component settings
                component_type = component.get("type", "")
                settings = component.get("settings", {})
                
                # Common settings for all components
                component["title"] = st.text_input("Component Title", value=component.get("title", ""), key=f"title_{i}")
                
                # Type-specific settings
                if component_type == "chart":
                    chart_type = component.get("chart_type", "bar")
                    chart_options = ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "area"]
                    
                    selected_chart = st.selectbox(
                        "Chart Type",
                        options=chart_options,
                        index=chart_options.index(chart_type) if chart_type in chart_options else 0,
                        key=f"chart_type_{i}"
                    )
                    component["chart_type"] = selected_chart
                    
                    # Column selectors for chart
                    x_col = settings.get("x_column")
                    x_options = [None] + st.session_state.data.columns.tolist()
                    x_index = x_options.index(x_col) if x_col in x_options else 0
                    
                    settings["x_column"] = st.selectbox(
                        "X-Axis Column",
                        options=x_options,
                        index=x_index,
                        key=f"x_col_{i}"
                    )
                    
                    # Y-axis for charts that need it
                    if selected_chart in ["bar", "line", "scatter", "box"]:
                        y_col = settings.get("y_column")
                        y_options = [None] + st.session_state.data.columns.tolist()
                        y_index = y_options.index(y_col) if y_col in y_options else 0
                        
                        settings["y_column"] = st.selectbox(
                            "Y-Axis Column",
                            options=y_options,
                            index=y_index,
                            key=f"y_col_{i}"
                        )
                    
                    # Color column
                    color_col = settings.get("color_column")
                    color_options = [None] + st.session_state.data.columns.tolist()
                    color_index = color_options.index(color_col) if color_col in color_options else 0
                    
                    settings["color_column"] = st.selectbox(
                        "Color By",
                        options=color_options,
                        index=color_index,
                        key=f"color_col_{i}"
                    )
                    
                    # Chart recommendations based on selected column
                    if settings["x_column"]:
                        st.subheader("Chart Recommendations")
                        recommendations = get_recommended_charts(st.session_state.data, settings["x_column"])
                        
                        for j, rec in enumerate(recommendations[:2]):  # Show top 2 recommendations
                            with st.container(border=True):
                                st.write(f"**{rec['title']}**")
                                st.write(f"Chart type: {rec['chart_type']}")
                                st.write(rec['description'])
                                
                                if st.button("Apply", key=f"apply_rec_{i}_{j}"):
                                    component["chart_type"] = rec["chart_type"]
                                    component["title"] = rec["title"]
                
                elif component_type == "table":
                    # Column selector for table
                    all_columns = st.session_state.data.columns.tolist()
                    selected_columns = settings.get("columns", all_columns)
                    
                    settings["columns"] = st.multiselect(
                        "Columns to Display",
                        options=all_columns,
                        default=selected_columns,
                        key=f"table_cols_{i}"
                    )
                    
                    # Page size
                    settings["page_size"] = st.number_input(
                        "Rows per page",
                        min_value=5,
                        max_value=100,
                        value=settings.get("page_size", 10),
                        step=5,
                        key=f"page_size_{i}"
                    )
                
                elif component_type == "metric":
                    # Column selector for metric
                    metric_col = settings.get("column")
                    metric_options = [None] + st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
                    metric_index = metric_options.index(metric_col) if metric_col in metric_options else 0
                    
                    settings["column"] = st.selectbox(
                        "Metric Column",
                        options=metric_options,
                        index=metric_index,
                        key=f"metric_col_{i}"
                    )
                    
                    # Aggregation method
                    agg_options = ["mean", "sum", "min", "max", "count"]
                    agg_index = agg_options.index(settings.get("aggregation", "mean"))
                    
                    settings["aggregation"] = st.selectbox(
                        "Aggregation Method",
                        options=agg_options,
                        index=agg_index,
                        key=f"agg_method_{i}"
                    )
                    
                    # Format string
                    settings["format"] = st.text_input(
                        "Format String (e.g., ${:.2f})",
                        value=settings.get("format", "{:.2f}"),
                        key=f"format_{i}"
                    )
                
                elif component_type == "text":
                    # Text content
                    settings["content"] = st.text_area(
                        "Text Content",
                        value=settings.get("content", ""),
                        height=150,
                        key=f"text_content_{i}"
                    )
                
                elif component_type == "filter":
                    st.info("Configure filter settings in the dashboard view")
                
                # Component actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Remove", key=f"remove_{i}", use_container_width=True):
                        remove_component(i)
                        st.rerun()
                
                with col2:
                    if st.button("Move Up", key=f"up_{i}", use_container_width=True, disabled=i==0):
                        move_component_up(i)
                        st.rerun()
                
                with col3:
                    if st.button("Move Down", key=f"down_{i}", use_container_width=True, disabled=i==len(st.session_state.dashboard_components)-1):
                        move_component_down(i)
                        st.rerun()

# Display dashboard preview at the bottom
if st.session_state.dashboard_components:
    st.divider()
    st.subheader("Dashboard Preview")
    st.caption("This is a simplified preview of how your dashboard will look.")
    
    # Create a more condensed preview
    preview_cols = st.columns(min(3, len(st.session_state.dashboard_components)))
    
    for i, component in enumerate(st.session_state.dashboard_components):
        with preview_cols[i % 3]:
            with st.container(border=True):
                st.write(f"**{component.get('title', 'Component')}**")
                st.caption(f"Type: {component.get('type', '').title()}")
