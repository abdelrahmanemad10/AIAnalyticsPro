import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from utils.ai_insights import (
    analyze_dataset,
    get_visualization_recommendations,
    generate_natural_language_query,
    suggest_data_transformations
)
from utils.visualization import create_chart
from utils.data_processor import transform_data

st.set_page_config(
    page_title="AI Insights | AI-Assisted Data Dashboard",
    page_icon="🤖",
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

# Initialize insights state
if 'general_insights' not in st.session_state:
    st.session_state.general_insights = None
if 'visualization_recommendations' not in st.session_state:
    st.session_state.visualization_recommendations = None
if 'data_transformations' not in st.session_state:
    st.session_state.data_transformations = None
if 'query_results' not in st.session_state:
    st.session_state.query_results = {}
if 'transformed_data' not in st.session_state:
    st.session_state.transformed_data = None

# Sidebar for controls
with st.sidebar:
    st.title("AI Insights")
    
    # Analysis type selector
    st.subheader("Analysis Type")
    analysis_type = st.radio(
        "Select analysis type",
        options=["general", "correlations", "outliers", "trends"],
        format_func=lambda x: x.title()
    )
    
    # Data scope
    st.subheader("Data Scope")
    
    # Column selection
    st.write("Select columns for analysis (optional)")
    selected_cols = st.multiselect(
        "Columns",
        options=df.columns.tolist(),
        default=[]
    )
    
    # Row sampling for large datasets
    max_rows = len(df)
    sample_size = max_rows
    
    if max_rows > 1000:
        st.write("Dataset is large. Consider using a sample for faster analysis.")
        use_sample = st.checkbox("Use sample", value=True)
        
        if use_sample:
            sample_size = st.slider(
                "Sample size",
                min_value=100,
                max_value=min(10000, max_rows),
                value=min(1000, max_rows),
                step=100
            )
    
    # Prepare the dataset for analysis
    if selected_cols and len(selected_cols) > 0:
        analysis_df = df[selected_cols].copy()
    else:
        analysis_df = df.copy()
    
    if sample_size < max_rows:
        analysis_df = analysis_df.sample(sample_size, random_state=42)
    
    st.write(f"Analyzing {len(analysis_df)} rows and {len(analysis_df.columns)} columns")
    
    # Generate insights button
    if st.button("Generate Insights", use_container_width=True):
        with st.spinner("Generating AI insights..."):
            try:
                st.session_state.general_insights = analyze_dataset(analysis_df, analysis_type)
                st.success("Insights generated!")
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
    
    # Get visualization recommendations
    if st.button("Recommend Visualizations", use_container_width=True):
        with st.spinner("Generating visualization recommendations..."):
            try:
                st.session_state.visualization_recommendations = get_visualization_recommendations(analysis_df)
                st.success("Recommendations generated!")
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
    
    # Suggest data transformations
    if st.button("Suggest Data Transformations", use_container_width=True):
        with st.spinner("Generating transformation suggestions..."):
            try:
                st.session_state.data_transformations = suggest_data_transformations(analysis_df)
                st.success("Transformation suggestions generated!")
            except Exception as e:
                st.error(f"Error generating transformation suggestions: {str(e)}")

# Main content
st.title("🤖 AI Insights")
st.write("Use AI to gain deeper insights into your data and discover patterns you might miss.")

# Tabs for different insight types
tab1, tab2, tab3, tab4 = st.tabs(["General Insights", "Visualization Recommendations", "Natural Language Queries", "Smart Transformations"])

with tab1:
    st.header("AI-Generated Insights")
    
    if st.session_state.general_insights is None:
        st.info("Click 'Generate Insights' in the sidebar to analyze your data.")
    else:
        insights = st.session_state.general_insights
        
        # Handle error case
        if "error" in insights:
            st.error(insights["error"])
        else:
            # Display insights based on analysis type
            if analysis_type == "general":
                # Summary section
                if "summary" in insights:
                    st.subheader("Data Summary")
                    st.write(insights["summary"])
                
                # Patterns section
                if "patterns" in insights:
                    st.subheader("Key Patterns")
                    patterns = insights["patterns"]
                    if isinstance(patterns, list):
                        for pattern in patterns:
                            st.write(f"• {pattern}")
                    else:
                        st.write(patterns)
                
                # Data quality section
                if "data_quality" in insights:
                    st.subheader("Data Quality Issues")
                    quality = insights["data_quality"]
                    if isinstance(quality, list):
                        for issue in quality:
                            st.write(f"• {issue}")
                    else:
                        st.write(quality)
                
                # Recommended visualizations
                if "recommended_visualizations" in insights:
                    st.subheader("Recommended Visualizations")
                    viz_recs = insights["recommended_visualizations"]
                    if isinstance(viz_recs, list):
                        for viz in viz_recs:
                            st.write(f"• {viz}")
                    else:
                        st.write(viz_recs)
                
                # Business insights
                if "business_insights" in insights:
                    st.subheader("Business Insights")
                    biz_insights = insights["business_insights"]
                    if isinstance(biz_insights, list):
                        for insight in biz_insights:
                            st.write(f"• {insight}")
                    else:
                        st.write(biz_insights)
            
            elif analysis_type == "correlations":
                # Strongest correlations
                if "strongest_correlations" in insights:
                    st.subheader("Strongest Correlations")
                    st.write(insights["strongest_correlations"])
                
                # Explanations
                if "explanations" in insights:
                    st.subheader("Explanation of Correlations")
                    explanations = insights["explanations"]
                    if isinstance(explanations, list):
                        for exp in explanations:
                            st.write(f"• {exp}")
                    else:
                        st.write(explanations)
                
                # Visualization recommendations
                if "visualization_recommendations" in insights:
                    st.subheader("Suggested Visualizations")
                    viz_recs = insights["visualization_recommendations"]
                    if isinstance(viz_recs, list):
                        for viz in viz_recs:
                            st.write(f"• {viz}")
                    else:
                        st.write(viz_recs)
                
                # Causation hypotheses
                if "causation_hypotheses" in insights:
                    st.subheader("Potential Causation Hypotheses")
                    st.write("*Note: Correlation does not imply causation*")
                    hypotheses = insights["causation_hypotheses"]
                    if isinstance(hypotheses, list):
                        for hyp in hypotheses:
                            st.write(f"• {hyp}")
                    else:
                        st.write(hypotheses)
            
            elif analysis_type == "outliers":
                # Outlier columns
                if "outlier_columns" in insights:
                    st.subheader("Columns with Potential Outliers")
                    outlier_cols = insights["outlier_columns"]
                    if isinstance(outlier_cols, list):
                        for col in outlier_cols:
                            st.write(f"• {col}")
                    else:
                        st.write(outlier_cols)
                
                # Detection methods
                if "detection_methods" in insights:
                    st.subheader("Recommended Detection Methods")
                    methods = insights["detection_methods"]
                    if isinstance(methods, list):
                        for method in methods:
                            st.write(f"• {method}")
                    else:
                        st.write(methods)
                
                # Impact analysis
                if "impact" in insights:
                    st.subheader("Impact of Outliers")
                    st.write(insights["impact"])
                
                # Handling suggestions
                if "handling_suggestions" in insights:
                    st.subheader("Suggestions for Handling Outliers")
                    suggestions = insights["handling_suggestions"]
                    if isinstance(suggestions, list):
                        for suggestion in suggestions:
                            st.write(f"• {suggestion}")
                    else:
                        st.write(suggestions)
            
            elif analysis_type == "trends":
                # Key trends
                if "key_trends" in insights:
                    st.subheader("Key Trends")
                    trends = insights["key_trends"]
                    if isinstance(trends, list):
                        for trend in trends:
                            st.write(f"• {trend}")
                    else:
                        st.write(trends)
                
                # Seasonal patterns
                if "seasonal_patterns" in insights:
                    st.subheader("Seasonal Patterns")
                    patterns = insights["seasonal_patterns"]
                    if isinstance(patterns, list):
                        for pattern in patterns:
                            st.write(f"• {pattern}")
                    else:
                        st.write(patterns)
                
                # Group differences
                if "group_differences" in insights:
                    st.subheader("Group Differences")
                    differences = insights["group_differences"]
                    if isinstance(differences, list):
                        for diff in differences:
                            st.write(f"• {diff}")
                    else:
                        st.write(differences)
                
                # Visualization recommendations
                if "visualization_recommendations" in insights:
                    st.subheader("Recommended Visualizations")
                    viz_recs = insights["visualization_recommendations"]
                    if isinstance(viz_recs, list):
                        for viz in viz_recs:
                            st.write(f"• {viz}")
                    else:
                        st.write(viz_recs)
            
            # Display any other sections that might be in the response
            for key, value in insights.items():
                if key not in ["summary", "patterns", "data_quality", "recommended_visualizations", 
                               "business_insights", "strongest_correlations", "explanations", 
                               "visualization_recommendations", "causation_hypotheses", "outlier_columns", 
                               "detection_methods", "impact", "handling_suggestions", "key_trends", 
                               "seasonal_patterns", "group_differences", "error"]:
                    st.subheader(key.replace("_", " ").title())
                    if isinstance(value, list):
                        for item in value:
                            st.write(f"• {item}")
                    else:
                        st.write(value)

with tab2:
    st.header("AI-Recommended Visualizations")
    
    if st.session_state.visualization_recommendations is None:
        st.info("Click 'Recommend Visualizations' in the sidebar to get visualization suggestions.")
    else:
        recommendations = st.session_state.visualization_recommendations
        
        # Handle error case
        if len(recommendations) == 1 and "error" in recommendations[0]:
            st.error(recommendations[0]["error"])
        else:
            for i, rec in enumerate(recommendations):
                with st.expander(f"Recommendation {i+1}: {rec.get('title', f'Chart {i+1}')}"):
                    # Display recommendation details
                    st.subheader(rec.get("title", f"Chart {i+1}"))
                    st.write(f"**Chart Type:** {rec.get('chart_type', 'N/A')}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**X-Axis:** {rec.get('x_column', 'N/A')}")
                    with col2:
                        if "y_column" in rec:
                            st.write(f"**Y-Axis:** {rec.get('y_column')}")
                    with col3:
                        if "color_column" in rec:
                            st.write(f"**Color:** {rec.get('color_column')}")
                    
                    st.write(f"**Description:** {rec.get('description', 'No description available')}")
                    
                    if "preparation" in rec and rec["preparation"]:
                        st.write(f"**Data Preparation:** {rec['preparation']}")
                    
                    # Try to create the chart
                    try:
                        x_col = rec.get("x_column")
                        y_col = rec.get("y_column")
                        chart_type = rec.get("chart_type")
                        color_col = rec.get("color_column")
                        title = rec.get("title")
                        
                        if x_col in df.columns:
                            if chart_type in ["scatter", "line", "bar", "box"] and y_col not in df.columns:
                                st.warning(f"Y-axis column '{y_col}' not found in data.")
                            else:
                                fig = create_chart(
                                    df, 
                                    chart_type=chart_type, 
                                    x_col=x_col, 
                                    y_col=y_col, 
                                    color_col=color_col, 
                                    title=title
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Column '{x_col}' not found in data.")
                    
                    except Exception as e:
                        st.error(f"Error creating chart: {str(e)}")
                        
                    # Button to add this chart to the dashboard
                    if st.button("Add to Dashboard", key=f"add_viz_{i}"):
                        if 'dashboard_components' not in st.session_state:
                            st.session_state.dashboard_components = []
                        
                        # Create chart component
                        chart_settings = {
                            "x_column": rec.get("x_column"),
                            "y_column": rec.get("y_column"),
                            "color_column": rec.get("color_column"),
                            "title": rec.get("title")
                        }
                        
                        from utils.dashboard_components import create_chart_component
                        component = create_chart_component(
                            chart_type=rec.get("chart_type", "bar"),
                            settings=chart_settings
                        )
                        
                        st.session_state.dashboard_components.append(component)
                        st.success("Chart added to dashboard!")

with tab3:
    st.header("Natural Language Queries")
    st.write("Ask questions about your data in plain English")
    
    # Text input for queries
    query = st.text_input(
        "Ask a question about your data",
        placeholder="e.g., What is the average sales by region?"
    )
    
    # Process the query
    if query:
        if query in st.session_state.query_results:
            # Use cached result
            query_result = st.session_state.query_results[query]
        else:
            # Process new query
            with st.spinner("Processing your query..."):
                try:
                    query_result = generate_natural_language_query(df, query)
                    st.session_state.query_results[query] = query_result
                except Exception as e:
                    query_result = {"error": str(e), "success": False}
                    st.session_state.query_results[query] = query_result
        
        # Display the result
        if query_result.get("success", False):
            st.subheader("Result")
            
            # Display explanation
            if "explanation" in query_result:
                st.write(query_result["explanation"])
            
            # Display code used
            with st.expander("Code Used"):
                st.code(query_result["code"], language="python")
            
            # Display the actual result
            result = query_result.get("result")
            if result is not None:
                if isinstance(result, (pd.DataFrame, dict)) or (isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict)):
                    try:
                        if isinstance(result, dict):
                            result_df = pd.DataFrame([result])
                        elif isinstance(result, list):
                            result_df = pd.DataFrame(result)
                        else:
                            result_df = result
                        
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Try to create a visualization if the result is tabular
                        if len(result_df) > 1 and len(result_df.columns) >= 2:
                            # Check if there are numeric columns
                            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
                            if numeric_cols:
                                st.subheader("Visualization")
                                
                                # Choose appropriate visualization based on data
                                if len(result_df) <= 20:  # For smaller results
                                    fig = create_chart(
                                        result_df,
                                        chart_type="bar",
                                        x_col=result_df.columns[0],
                                        y_col=numeric_cols[0],
                                        title=f"Query Result: {query}"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:  # For larger results
                                    fig = create_chart(
                                        result_df,
                                        chart_type="line",
                                        x_col=result_df.columns[0],
                                        y_col=numeric_cols[0],
                                        title=f"Query Result: {query}"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    except Exception as viz_error:
                        st.warning(f"Could not create visualization: {str(viz_error)}")
                else:
                    # For single values or lists of single values
                    st.write(result)
        else:
            # Display error
            st.error(query_result.get("error", "Unknown error processing your query"))
    
    # Query history
    if st.session_state.query_results:
        st.subheader("Query History")
        
        for past_query in list(st.session_state.query_results.keys())[-5:]:  # Show last 5 queries
            if st.button(past_query, key=f"hist_{past_query}"):
                # Rerun with this query
                st.session_state.rerun_query = past_query
                st.rerun()

with tab4:
    st.header("Smart Data Transformations")
    
    if st.session_state.data_transformations is None:
        st.info("Click 'Suggest Data Transformations' in the sidebar to get transformation suggestions.")
    else:
        transformations = st.session_state.data_transformations
        
        # Handle error case
        if len(transformations) == 1 and "error" in transformations[0]:
            st.error(transformations[0]["error"])
        else:
            # Track applied transformations
            if 'applied_transformations' not in st.session_state:
                st.session_state.applied_transformations = []
            
            st.write("Select transformations to apply to your data:")
            
            for i, transform in enumerate(transformations):
                with st.expander(f"Transformation {i+1}: {transform.get('type', 'Unknown')}"):
                    # Display transformation details
                    st.subheader(transform.get("type", "Transformation").title())
                    
                    if "columns" in transform:
                        cols = transform["columns"]
                        if isinstance(cols, list):
                            st.write(f"**Applies to:** {', '.join(cols)}")
                        else:
                            st.write(f"**Applies to:** {cols}")
                    
                    if "rationale" in transform:
                        st.write(f"**Rationale:** {transform['rationale']}")
                    
                    if "code" in transform:
                        with st.expander("Python Code"):
                            st.code(transform["code"], language="python")
                    
                    # Button to apply this transformation
                    if st.button("Apply Transformation", key=f"apply_trans_{i}"):
                        # Add to applied transformations
                        st.session_state.applied_transformations.append(transform)
                        
                        # Generate transformation operation
                        operation = None
                        transform_type = transform.get("type", "").lower()
                        
                        if "missing" in transform_type:
                            # Handle missing values
                            if "code" in transform and "fillna" in transform["code"]:
                                # Extract column and fill value from code
                                import re
                                col_match = re.search(r'df\[[\'"](.*?)[\'"]\]', transform["code"])
                                if col_match:
                                    col = col_match.group(1)
                                    method = "mean" if "mean" in transform["code"] else "median" if "median" in transform["code"] else "0"
                                    
                                    operation = {
                                        "operation": "fill_na",
                                        "column": col,
                                        "method": method
                                    }
                        
                        elif "outlier" in transform_type:
                            # Handle outliers
                            if "columns" in transform and isinstance(transform["columns"], list):
                                col = transform["columns"][0]
                                
                                operation = {
                                    "operation": "remove_outliers",
                                    "column": col,
                                    "method": "iqr"  # Using IQR method as default
                                }
                        
                        elif "feature" in transform_type or "new column" in transform_type:
                            # Feature engineering / new column
                            if "code" in transform:
                                import re
                                new_col_match = re.search(r'df\[[\'"](.*?)[\'"]\]\s*=', transform["code"])
                                expr_match = re.search(r'=\s*(.*?)$', transform["code"], re.MULTILINE)
                                
                                if new_col_match and expr_match:
                                    new_col = new_col_match.group(1)
                                    expr = expr_match.group(1).strip()
                                    
                                    operation = {
                                        "operation": "new_column",
                                        "column": new_col,
                                        "expression": expr
                                    }
                        
                        elif "normalize" in transform_type or "scale" in transform_type:
                            # Normalization/scaling
                            if "columns" in transform:
                                cols = transform["columns"]
                                if isinstance(cols, list):
                                    col = cols[0]
                                else:
                                    col = cols
                                
                                operation = {
                                    "operation": "normalize",
                                    "column": col,
                                    "method": "minmax"  # Using min-max scaling as default
                                }
                        
                        # Add any transformation we could parse
                        if operation:
                            if 'transformations' not in st.session_state:
                                st.session_state.transformations = []
                            
                            st.session_state.transformations.append(operation)
                        
                        st.success("Transformation added!")
            
            # Apply all button 
            if st.button("Apply All Selected Transformations", use_container_width=True):
                try:
                    operations = []
                    
                    # Collect all operations that we've been able to parse
                    if 'transformations' in st.session_state and st.session_state.transformations:
                        operations = st.session_state.transformations
                    
                    # Apply the transformations
                    if operations:
                        with st.spinner("Applying transformations..."):
                            transformed_df = transform_data(df, operations)
                            st.session_state.transformed_data = transformed_df
                            st.success(f"Successfully applied {len(operations)} transformations!")
                    else:
                        st.warning("No transformations to apply.")
                
                except Exception as e:
                    st.error(f"Error applying transformations: {str(e)}")
            
            # Show transformed data if available
            if st.session_state.transformed_data is not None:
                st.subheader("Transformed Data Preview")
                st.dataframe(st.session_state.transformed_data.head(10), use_container_width=True)
                
                # Use transformed data option
                if st.button("Use Transformed Data"):
                    st.session_state.data = st.session_state.transformed_data
                    st.success("Main dataset updated with transformed data!")
                    st.rerun()

# Show API Key warning
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.sidebar.warning(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable to use AI features."
    )
