import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import get_column_stats, generate_data_summary, transform_data
from utils.visualization import (
    get_recommended_charts,
    get_bivariate_chart_recommendations,
    display_chart_with_controls,
    get_correlation_matrix,
    create_chart
)

st.set_page_config(
    page_title="Data Explorer | AI-Assisted Data Dashboard",
    page_icon="🔍",
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

# Initialize exploration states
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = df.copy()
if 'transformed_data' not in st.session_state:
    st.session_state.transformed_data = None
if 'transformations' not in st.session_state:
    st.session_state.transformations = []

# Sidebar for exploration controls
with st.sidebar:
    st.title("Data Explorer")
    
    # Column selection
    st.subheader("Column Selection")
    
    if st.button("Select All Columns"):
        st.session_state.selected_columns = df.columns.tolist()
    
    if st.button("Clear Selection"):
        st.session_state.selected_columns = []
    
    st.session_state.selected_columns = st.multiselect(
        "Select columns to explore",
        options=df.columns.tolist(),
        default=st.session_state.selected_columns
    )
    
    # Data filtering
    st.subheader("Data Filtering")
    
    if st.session_state.selected_columns:
        # Create filters for selected columns
        filters_applied = False
        filtered_df = df.copy()
        
        for col in st.session_state.selected_columns:
            if col in df.columns:
                # Determine column type
                if np.issubdtype(df[col].dtype, np.number):
                    # Numeric filter
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    
                    if min_val != max_val:
                        filter_range = st.slider(
                            f"Filter by {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"filter_{col}"
                        )
                        
                        if filter_range != (min_val, max_val):
                            filtered_df = filtered_df[(filtered_df[col] >= filter_range[0]) & 
                                                    (filtered_df[col] <= filter_range[1])]
                            filters_applied = True
                
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    # Date filter
                    min_date = df[col].min().date()
                    max_date = df[col].max().date()
                    
                    if min_date != max_date:
                        start_date = st.date_input(
                            f"Start date for {col}",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key=f"start_{col}"
                        )
                        
                        end_date = st.date_input(
                            f"End date for {col}",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key=f"end_{col}"
                        )
                        
                        filtered_df = filtered_df[(filtered_df[col].dt.date >= start_date) & 
                                                (filtered_df[col].dt.date <= end_date)]
                        filters_applied = True
                
                else:
                    # Categorical filter
                    unique_values = df[col].dropna().unique().tolist()
                    
                    if len(unique_values) <= 25:  # Only show for reasonable number of categories
                        selected_values = st.multiselect(
                            f"Filter by {col}",
                            options=unique_values,
                            default=unique_values,
                            key=f"cat_{col}"
                        )
                        
                        if len(selected_values) < len(unique_values):
                            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                            filters_applied = True
        
        if filters_applied:
            st.session_state.filtered_data = filtered_df
        else:
            st.session_state.filtered_data = df.copy()
            
        # Show filter info
        st.info(f"Showing {len(filtered_df)} of {len(df)} rows")
    
    # Data transformation
    st.subheader("Data Transformation")
    
    transform_type = st.selectbox(
        "Add transformation",
        options=["None", "Filter", "Sort", "Group By", "New Column", "Rename", "Drop"]
    )
    
    if transform_type != "None":
        if transform_type == "Filter":
            col = st.selectbox("Column", options=df.columns.tolist(), key="filter_col")
            condition = st.selectbox(
                "Condition", 
                options=["equals", "not_equals", "greater_than", "less_than", "contains", "in"],
                key="filter_condition"
            )
            value = st.text_input("Value", key="filter_value")
            
            if st.button("Add Filter"):
                st.session_state.transformations.append({
                    "operation": "filter",
                    "column": col,
                    "condition": condition,
                    "value": value
                })
                st.rerun()
        
        elif transform_type == "Sort":
            col = st.selectbox("Column", options=df.columns.tolist(), key="sort_col")
            ascending = st.checkbox("Ascending", value=True, key="sort_ascending")
            
            if st.button("Add Sort"):
                st.session_state.transformations.append({
                    "operation": "sort",
                    "column": col,
                    "ascending": ascending
                })
                st.rerun()
        
        elif transform_type == "Group By":
            cols = st.multiselect("Group By Columns", options=df.columns.tolist(), key="group_cols")
            agg_col = st.selectbox(
                "Aggregation Column", 
                options=[c for c in df.columns if c not in cols],
                key="agg_col"
            )
            agg_func = st.selectbox(
                "Aggregation Function", 
                options=["mean", "sum", "count", "min", "max"],
                key="agg_func"
            )
            
            if st.button("Add Group By"):
                st.session_state.transformations.append({
                    "operation": "group_by",
                    "columns": cols,
                    "aggregation": {agg_col: agg_func}
                })
                st.rerun()
        
        elif transform_type == "New Column":
            col_name = st.text_input("New Column Name", key="new_col_name")
            expression = st.text_input("Expression (e.g., column_a * column_b)", key="new_col_expr")
            
            if st.button("Add New Column"):
                st.session_state.transformations.append({
                    "operation": "new_column",
                    "column": col_name,
                    "expression": expression
                })
                st.rerun()
        
        elif transform_type == "Rename":
            old_col = st.selectbox("Column to Rename", options=df.columns.tolist(), key="rename_col")
            new_name = st.text_input("New Name", key="new_name")
            
            if st.button("Add Rename"):
                st.session_state.transformations.append({
                    "operation": "rename",
                    "mapping": {old_col: new_name}
                })
                st.rerun()
        
        elif transform_type == "Drop":
            cols = st.multiselect("Columns to Drop", options=df.columns.tolist(), key="drop_cols")
            
            if st.button("Add Drop"):
                st.session_state.transformations.append({
                    "operation": "drop",
                    "columns": cols
                })
                st.rerun()
    
    # Apply transformations
    if st.session_state.transformations:
        st.divider()
        
        # Display current transformations
        st.subheader("Current Transformations")
        for i, transform in enumerate(st.session_state.transformations):
            op = transform.get("operation", "")
            if op == "filter":
                st.write(f"{i+1}. Filter: {transform.get('column')} {transform.get('condition')} {transform.get('value')}")
            elif op == "sort":
                direction = "Ascending" if transform.get("ascending", True) else "Descending"
                st.write(f"{i+1}. Sort: {transform.get('column')} ({direction})")
            elif op == "group_by":
                cols = ", ".join(transform.get("columns", []))
                agg = transform.get("aggregation", {})
                agg_str = ", ".join([f"{k}({v})" for k, v in agg.items()])
                st.write(f"{i+1}. Group By: {cols} - {agg_str}")
            elif op == "new_column":
                st.write(f"{i+1}. New Column: {transform.get('column')} = {transform.get('expression')}")
            elif op == "rename":
                mapping = transform.get("mapping", {})
                for old, new in mapping.items():
                    st.write(f"{i+1}. Rename: {old} → {new}")
            elif op == "drop":
                cols = ", ".join(transform.get("columns", []))
                st.write(f"{i+1}. Drop: {cols}")
        
        # Apply transformations button
        if st.button("Apply Transformations"):
            try:
                st.session_state.transformed_data = transform_data(st.session_state.filtered_data, st.session_state.transformations)
                st.success("Transformations applied successfully!")
            except Exception as e:
                st.error(f"Error applying transformations: {str(e)}")
        
        # Clear transformations button
        if st.button("Clear Transformations"):
            st.session_state.transformations = []
            st.session_state.transformed_data = None
            st.rerun()

# Main content
st.title("🔍 Data Explorer")
st.write("Explore your dataset and discover insights through visualizations and statistical analysis.")

# Determine which data to show
display_df = st.session_state.transformed_data if st.session_state.transformed_data is not None else st.session_state.filtered_data

# Overview tab and visualization tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Univariate Analysis", "Bivariate Analysis", "Correlations"])

with tab1:
    st.header("Data Overview")
    
    # Data summary
    with st.expander("Dataset Summary", expanded=True):
        # Basic info
        st.subheader("Basic Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Rows", len(display_df))
        
        with info_col2:
            st.metric("Columns", len(display_df.columns))
        
        with info_col3:
            numeric_cols = display_df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        
        # Generate data summary
        st.subheader("Data Summary")
        summary_text = generate_data_summary(display_df)
        st.text(summary_text)
        
        # Column data types
        st.subheader("Column Data Types")
        dtypes_df = pd.DataFrame(display_df.dtypes, columns=["Data Type"])
        st.dataframe(dtypes_df)
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(display_df.head(10), use_container_width=True)
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_df = pd.DataFrame(display_df.isnull().sum(), columns=["Missing Values"])
    missing_df["Percentage"] = (missing_df["Missing Values"] / len(display_df) * 100).round(2)
    missing_df = missing_df[missing_df["Missing Values"] > 0]
    
    if not missing_df.empty:
        st.bar_chart(missing_df["Percentage"])
        st.dataframe(missing_df)
    else:
        st.success("No missing values in the dataset!")

with tab2:
    st.header("Univariate Analysis")
    st.write("Analyze individual variables and their distributions.")
    
    # Column selector
    col_options = display_df.columns.tolist()
    selected_col = st.selectbox("Select a column to analyze", options=col_options)
    
    if selected_col:
        col_data = display_df[selected_col]
        col_type = display_df[selected_col].dtype
        
        # Column statistics
        st.subheader(f"Statistics for {selected_col}")
        
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.write("**Basic Statistics:**")
            if np.issubdtype(col_type, np.number):
                stats = {
                    "Mean": col_data.mean(),
                    "Median": col_data.median(),
                    "Std Dev": col_data.std(),
                    "Min": col_data.min(),
                    "Max": col_data.max()
                }
                
                for stat, value in stats.items():
                    st.write(f"**{stat}:** {value:.4f}")
            else:
                st.write(f"**Unique Values:** {col_data.nunique()}")
                st.write(f"**Most Common:** {col_data.value_counts().index[0] if not col_data.value_counts().empty else 'N/A'}")
                st.write(f"**Least Common:** {col_data.value_counts().index[-1] if not col_data.value_counts().empty else 'N/A'}")
        
        with stat_col2:
            st.write("**Missing Values:**")
            missing_count = col_data.isnull().sum()
            missing_percent = (missing_count / len(col_data) * 100)
            
            st.write(f"**Count:** {missing_count}")
            st.write(f"**Percentage:** {missing_percent:.2f}%")
            
            if np.issubdtype(col_type, np.number):
                st.write("**Quartiles:**")
                quartiles = col_data.quantile([0.25, 0.5, 0.75]).to_dict()
                for q, value in quartiles.items():
                    st.write(f"**Q{int(q*4)}:** {value:.4f}")
        
        # Chart recommendations
        st.subheader("Recommended Visualizations")
        recommendations = get_recommended_charts(display_df, selected_col)
        
        for i, rec in enumerate(recommendations[:3]):  # Show top 3 recommendations
            chart_type = rec["chart_type"]
            title = rec["title"]
            
            st.write(f"**{i+1}. {title}**")
            st.write(rec["description"])
            
            try:
                fig = create_chart(
                    display_df, 
                    chart_type=chart_type, 
                    x_col=selected_col, 
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
        
        # Value counts for categorical data
        if not np.issubdtype(col_type, np.number) and col_data.nunique() < 50:
            st.subheader("Value Counts")
            value_counts = col_data.value_counts().reset_index()
            value_counts.columns = [selected_col, "Count"]
            
            fig = create_chart(
                value_counts,
                chart_type="bar",
                x_col=selected_col,
                y_col="Count",
                title=f"Count of {selected_col} Values"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Bivariate Analysis")
    st.write("Analyze relationships between two variables.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Select X-axis column", options=display_df.columns.tolist(), key="x_col")
    
    with col2:
        y_col = st.selectbox("Select Y-axis column", options=display_df.columns.tolist(), key="y_col")
    
    if x_col and y_col:
        # Chart recommendations
        st.subheader("Recommended Visualizations")
        bivariate_recs = get_bivariate_chart_recommendations(display_df, x_col, y_col)
        
        for i, rec in enumerate(bivariate_recs[:2]):  # Show top 2 recommendations
            chart_type = rec["chart_type"]
            title = rec["title"]
            
            st.write(f"**{i+1}. {title}**")
            st.write(rec["description"])
            
            fig = display_chart_with_controls(
                display_df, 
                chart_type=chart_type, 
                x_col=x_col, 
                y_col=y_col,
                default_title=title
            )
        
        # Additional visualizations based on data types
        x_type = display_df[x_col].dtype
        y_type = display_df[y_col].dtype
        
        # For two numeric columns, add a scatter plot with regression line
        if np.issubdtype(x_type, np.number) and np.issubdtype(y_type, np.number):
            st.subheader("Scatter Plot with Trend Line")
            
            fig = create_chart(
                display_df,
                chart_type="scatter",
                x_col=x_col,
                y_col=y_col,
                title=f"{y_col} vs {x_col} with Trend Line",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # For categorical vs numeric, add a box plot and bar chart
        elif (not np.issubdtype(x_type, np.number) and np.issubdtype(y_type, np.number)):
            st.subheader("Box Plot by Category")
            
            fig = create_chart(
                display_df,
                chart_type="box",
                x_col=x_col,
                y_col=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # For numeric vs categorical, add a box plot (reversed)
        elif (np.issubdtype(x_type, np.number) and not np.issubdtype(y_type, np.number)):
            st.subheader("Box Plot by Category")
            
            fig = create_chart(
                display_df,
                chart_type="box",
                x_col=y_col,
                y_col=x_col,
                title=f"Distribution of {x_col} by {y_col}"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Correlation Analysis")
    st.write("Analyze correlations between numerical variables.")
    
    # Get numeric columns
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
    else:
        # Correlation matrix
        st.subheader("Correlation Matrix")
        
        # Select columns for correlation
        selected_corr_cols = st.multiselect(
            "Select columns for correlation analysis",
            options=numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )
        
        if selected_corr_cols and len(selected_corr_cols) >= 2:
            corr_matrix = get_correlation_matrix(display_df, selected_corr_cols)
            
            # Display as heatmap
            fig = create_chart(
                display_df, 
                chart_type="heatmap", 
                title="Correlation Matrix",
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display as table
            st.subheader("Correlation Values")
            st.dataframe(corr_matrix.style.highlight_max(axis=None).highlight_min(axis=None), use_container_width=True)
            
            # Strong correlations
            st.subheader("Strong Correlations")
            
            # Create a DataFrame of correlations
            corr_pairs = []
            for i in range(len(selected_corr_cols)):
                for j in range(i+1, len(selected_corr_cols)):
                    col1 = selected_corr_cols[i]
                    col2 = selected_corr_cols[j]
                    corr_value = corr_matrix.loc[col1, col2]
                    
                    corr_pairs.append({
                        "Variable 1": col1,
                        "Variable 2": col2,
                        "Correlation": corr_value
                    })
            
            if corr_pairs:
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.sort_values("Correlation", key=abs, ascending=False)
                
                # Display top correlations
                st.dataframe(corr_df, use_container_width=True)
                
                # Show scatter plot for top correlation
                if not corr_df.empty:
                    top_pair = corr_df.iloc[0]
                    var1 = top_pair["Variable 1"]
                    var2 = top_pair["Variable 2"]
                    corr = top_pair["Correlation"]
                    
                    st.subheader(f"Scatter Plot: {var1} vs {var2} (r = {corr:.4f})")
                    
                    fig = create_chart(
                        display_df,
                        chart_type="scatter",
                        x_col=var1,
                        y_col=var2,
                        title=f"{var2} vs {var1} (r = {corr:.4f})",
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
