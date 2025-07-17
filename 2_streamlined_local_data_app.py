import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Local Data Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Local Data Analysis Dashboard")
st.markdown("Upload your data files and analyze them locally on your machine")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Upload", "Data Analysis"])

if page == "Data Upload":
    st.header("Upload Your Data")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file to analyze"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            
            # Store in session state
            st.session_state['data'] = df
            
            st.success(f"File uploaded successfully! Shape: {df.shape}")
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Preview data
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("Please upload a CSV file to get started")

elif page == "Data Analysis":
    st.header("Data Analysis")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Basic statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Missing values analysis
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            st.write("Missing values by column:")
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Percentage': (missing_data.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found!")
        
        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()
            st.dataframe(corr_matrix, use_container_width=True)
        
        # Data type analysis
        st.subheader("Data Types Summary")
        dtype_summary = df.dtypes.value_counts()
        dtype_df = pd.DataFrame({
            'Data Type': dtype_summary.index,
            'Count': dtype_summary.values
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Unique values analysis
        st.subheader("Unique Values per Column")
        unique_counts = df.nunique().sort_values(ascending=False)
        unique_df = pd.DataFrame({
            'Column': unique_counts.index,
            'Unique Values': unique_counts.values,
            'Percentage Unique': (unique_counts.values / len(df) * 100).round(2)
        })
        st.dataframe(unique_df, use_container_width=True)
        
        # Data filtering and viewing
        st.subheader("Filter and View Data")
        
        # Select columns to display
        display_cols = st.multiselect(
            "Select columns to display",
            df.columns.tolist(),
            default=df.columns.tolist()[:5]
        )
        
        if display_cols:
            # Filter by numeric columns
            numeric_cols_available = [col for col in display_cols if col in numeric_cols]
            if numeric_cols_available:
                selected_numeric_col = st.selectbox("Filter by numeric column", ["None"] + numeric_cols_available)
                
                if selected_numeric_col != "None":
                    min_val, max_val = st.slider(
                        f"Range for {selected_numeric_col}",
                        float(df[selected_numeric_col].min()),
                        float(df[selected_numeric_col].max()),
                        (float(df[selected_numeric_col].min()), float(df[selected_numeric_col].max()))
                    )
                    filtered_df = df[(df[selected_numeric_col] >= min_val) & (df[selected_numeric_col] <= max_val)]
                else:
                    filtered_df = df
            else:
                filtered_df = df
            
            # Display filtered results
            st.write(f"Showing {len(filtered_df)} rows out of {len(df)} total rows")
            st.dataframe(filtered_df[display_cols], use_container_width=True)
        
        # Value counts for categorical columns
        st.subheader("Value Counts for Categorical Columns")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            selected_cat_col = st.selectbox("Select categorical column", categorical_cols)
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts().head(20)
                value_counts_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(df) * 100).round(2)
                })
                st.dataframe(value_counts_df, use_container_width=True)
        else:
            st.info("No categorical columns found in the dataset")
        
        # Numeric column statistics
        if len(numeric_cols) > 0:
            st.subheader("Individual Numeric Column Analysis")
            selected_num_col = st.selectbox("Select numeric column for detailed analysis", numeric_cols)
            
            if selected_num_col:
                col_data = df[selected_num_col]
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{col_data.mean():.2f}")
                with col2:
                    st.metric("Median", f"{col_data.median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{col_data.std():.2f}")
                with col4:
                    st.metric("Range", f"{col_data.max() - col_data.min():.2f}")
                
                # Additional statistics
                st.write("**Percentiles:**")
                percentiles = col_data.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                perc_df = pd.DataFrame({
                    'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
                    'Value': [f"{val:.2f}" for val in percentiles.values]
                })
                st.dataframe(perc_df, use_container_width=True)
        
    else:
        st.warning("Please upload data first in the 'Data Upload' section")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Privacy Note**: All data processing happens locally on your machine. No data is sent to external servers.")