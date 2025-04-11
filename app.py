import streamlit as st
import os
import pandas as pd
from utils.data_processor import load_sample_data, process_uploaded_file
from utils.dashboard_components import show_welcome_page

# Page configuration
st.set_page_config(
    page_title="AI-Assisted Data Dashboard Builder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'dashboard_components' not in st.session_state:
    st.session_state.dashboard_components = []
if 'current_template' not in st.session_state:
    st.session_state.current_template = "blank"

# Sidebar for data upload and navigation
with st.sidebar:
    st.title("📊 Data Dashboard")
    
    # File upload section
    st.header("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=["csv", "xlsx", "xls", "json", "tsv"],
        help="Supported formats: CSV, Excel, JSON, TSV"
    )
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            data, filename = process_uploaded_file(uploaded_file)
            st.session_state.data = data
            st.session_state.filename = filename
            st.success(f"Successfully loaded: {filename}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Sample data option
    if st.button("Use Sample Data"):
        try:
            st.session_state.data, st.session_state.filename = load_sample_data()
            st.success(f"Loaded sample data: {st.session_state.filename}")
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            
    # Only show navigation when data is loaded
    if st.session_state.data is not None:
        st.divider()
        st.header("Navigation")
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/dashboard_builder.py", label="Dashboard Builder", icon="🔨")
        st.page_link("pages/data_explorer.py", label="Data Explorer", icon="🔍")
        st.page_link("pages/ai_insights.py", label="AI Insights", icon="🤖") 
        st.page_link("pages/report_generator.py", label="Report Generator", icon="📄")

# Main content
if st.session_state.data is None:
    show_welcome_page()
else:
    st.title("Data Overview")
    
    data_info_col, preview_col = st.columns([1, 2])
    
    with data_info_col:
        st.subheader("Dataset Information")
        st.write(f"**Filename:** {st.session_state.filename}")
        st.write(f"**Rows:** {len(st.session_state.data)}")
        st.write(f"**Columns:** {len(st.session_state.data.columns)}")
        
        # Display data types
        st.subheader("Column Data Types")
        dtypes_df = pd.DataFrame(
            st.session_state.data.dtypes, 
            columns=['Data Type']
        )
        st.dataframe(dtypes_df)
        
        # Display basic statistics
        st.subheader("Missing Values")
        missing_values = pd.DataFrame(
            st.session_state.data.isnull().sum(), 
            columns=['Missing Values']
        )
        st.dataframe(missing_values)
        
    with preview_col:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Quick actions
        st.subheader("Quick Actions")
        quick_action_cols = st.columns(3)
        
        with quick_action_cols[0]:
            if st.button("Create Quick Dashboard", use_container_width=True):
                st.switch_page("pages/dashboard_builder.py")
                
        with quick_action_cols[1]:
            if st.button("Explore Data", use_container_width=True):
                st.switch_page("pages/data_explorer.py")
                
        with quick_action_cols[2]:
            if st.button("Get AI Insights", use_container_width=True):
                st.switch_page("pages/ai_insights.py")

# Footer
st.divider()
st.caption("AI-Assisted Data Dashboard Builder | Created with Streamlit")
