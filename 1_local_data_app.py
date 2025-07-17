import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Local Data Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Local Data Analysis Dashboard")
st.markdown("Upload your data files and analyze them locally on your machine")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Upload", "Data Analysis", "Visualization", "Sample Data"])

if page == "Data Upload":
    st.header("📁 Upload Your Data")
    
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
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
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
        st.info("👆 Please upload a CSV file to get started")

elif page == "Data Analysis":
    st.header("🔍 Data Analysis")
    
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
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title="Missing Values by Column"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values found!")
        
        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data filtering
        st.subheader("Filter Data")
        
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
            
            st.dataframe(filtered_df[display_cols], use_container_width=True)
        
    else:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section")

elif page == "Visualization":
    st.header("📈 Data Visualization")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select chart type",
            ["Histogram", "Scatter Plot", "Line Chart", "Box Plot", "Bar Chart"]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if chart_type == "Histogram":
            if numeric_cols:
                col = st.selectbox("Select column for histogram", numeric_cols)
                bins = st.slider("Number of bins", 10, 100, 30)
                
                fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for histogram")
        
        elif chart_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col])
                
                color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                
                if color_col == "None":
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for scatter plot")
        
        elif chart_type == "Line Chart":
            if numeric_cols:
                y_cols = st.multiselect("Select columns for Y-axis", numeric_cols)
                if y_cols:
                    fig = go.Figure()
                    for col in y_cols:
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df[col],
                            mode='lines',
                            name=col
                        ))
                    fig.update_layout(title="Line Chart", xaxis_title="Index", yaxis_title="Value")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Box Plot":
            if numeric_cols:
                col = st.selectbox("Select column for box plot", numeric_cols)
                group_by = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
                
                if group_by == "None":
                    fig = px.box(df, y=col, title=f"Box Plot of {col}")
                else:
                    fig = px.box(df, x=group_by, y=col, title=f"Box Plot of {col} by {group_by}")
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Bar Chart":
            if categorical_cols:
                col = st.selectbox("Select categorical column", categorical_cols)
                
                # Count values
                value_counts = df[col].value_counts().head(20)  # Top 20 values
                
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Value Counts for {col}"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical columns available for bar chart")
    
    else:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section")

elif page == "Sample Data":
    st.header("🎲 Sample Data Generator")
    st.markdown("Generate sample data to test the app functionality")
    
    data_type = st.selectbox(
        "Select sample data type",
        ["Sales Data", "Time Series", "Customer Data", "Random Dataset"]
    )
    
    if st.button("Generate Sample Data"):
        if data_type == "Sales Data":
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=1000, freq='D')
            df = pd.DataFrame({
                'date': np.random.choice(dates, 500),
                'product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 500),
                'sales': np.random.normal(1000, 300, 500),
                'quantity': np.random.poisson(10, 500),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 500)
            })
            df['sales'] = np.abs(df['sales'])  # Ensure positive sales
            
        elif data_type == "Time Series":
            dates = pd.date_range('2023-01-01', periods=365, freq='D')
            trend = np.linspace(100, 200, 365)
            seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 365)
            noise = np.random.normal(0, 10, 365)
            df = pd.DataFrame({
                'date': dates,
                'value': trend + seasonal + noise,
                'category': np.random.choice(['A', 'B'], 365)
            })
            
        elif data_type == "Customer Data":
            np.random.seed(42)
            df = pd.DataFrame({
                'customer_id': range(1, 301),
                'age': np.random.randint(18, 80, 300),
                'income': np.random.normal(50000, 20000, 300),
                'purchases': np.random.poisson(5, 300),
                'satisfaction': np.random.uniform(1, 5, 300),
                'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 300)
            })
            df['income'] = np.abs(df['income'])  # Ensure positive income
            
        else:  # Random Dataset
            np.random.seed(42)
            df = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 200),
                'feature2': np.random.exponential(2, 200),
                'feature3': np.random.uniform(-10, 10, 200),
                'category': np.random.choice(['X', 'Y', 'Z'], 200),
                'target': np.random.binomial(1, 0.3, 200)
            })
        
        st.session_state['data'] = df
        st.success(f"✅ Generated {data_type} with {df.shape[0]} rows and {df.shape[1]} columns")
        st.dataframe(df.head(), use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🔒 Privacy Note**: All data processing happens locally on your machine. No data is sent to external servers.")