import streamlit as st
import pandas as pd
import numpy as np
import json
import openai
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

def display_json_hierarchical(data, level=0):
    """Display JSON data in a hierarchical, expandable format"""
    if not isinstance(data, dict):
        st.write(str(data))
        return
    
    for key, value in data.items():
        if isinstance(value, dict) and len(value) > 0:
            if level == 0:
                # Top level - use expander
                with st.expander(f"{key}", expanded=True):
                    display_json_content(value, level + 1)
            else:
                # Nested levels - use formatted text with indentation
                indent = "  " * (level - 1)
                st.markdown(f"**{indent}📁 {key}:**")
                display_json_content(value, level + 1)
        elif isinstance(value, list):
            if level == 0:
                # Top level - use expander
                with st.expander(f"{key} ({len(value)} items)", expanded=False):
                    display_json_list(value, level + 1)
            else:
                # Nested levels - use formatted text
                indent = "  " * (level - 1)
                st.markdown(f"**{indent}📋 {key} ({len(value)} items):**")
                display_json_list(value, level + 1)
        else:
            # Simple key-value pair
            indent = "  " * level
            if value is None:
                st.markdown(f"{indent}**{key}:** *N/A*")
            else:
                st.markdown(f"{indent}**{key}:** {value}")

def display_json_content(data, level):
    """Display dictionary content with proper indentation"""
    for key, value in data.items():
        if isinstance(value, dict) and len(value) > 0:
            indent = "  " * (level - 1)
            st.markdown(f"**{indent}📁 {key}:**")
            display_json_content(value, level + 1)
        elif isinstance(value, list):
            indent = "  " * (level - 1)
            st.markdown(f"**{indent}📋 {key} ({len(value)} items):**")
            display_json_list(value, level + 1)
        else:
            indent = "  " * level
            if value is None:
                st.markdown(f"{indent}**{key}:** *N/A*")
            else:
                st.markdown(f"{indent}**{key}:** {value}")

def display_json_list(items, level):
    """Display list items with proper indentation"""
    for i, item in enumerate(items):
        indent = "  " * level
        if isinstance(item, dict):
            st.markdown(f"{indent}**Item {i + 1}:**")
            display_json_content(item, level + 1)
        else:
            st.markdown(f"{indent}**Item {i + 1}:** {item}")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Upload", "Data Analysis", "Individual Records", "JSON Record Viewer", "AI Analysis - Individual", "AI Analysis - Entity"])

if page == "Data Upload":
    st.header("Upload Your Data")
    
    # API Key section at the top
    st.subheader("OpenAI API Configuration")
    api_key_input = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to enable AI analysis features",
        placeholder="sk-..."
    )
    
    col_api1, col_api2 = st.columns([1, 1])
    with col_api1:
        if st.button("Save API Key", use_container_width=True):
            if api_key_input:
                st.session_state['openai_api_key'] = api_key_input
                st.success("API Key saved successfully!")
            else:
                st.error("Please enter a valid API key")
    
    with col_api2:
        if st.button("Test API Key", use_container_width=True):
            if 'openai_api_key' in st.session_state:
                try:
                    # Test the API key with a simple call
                    test_client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
                    test_response = test_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5
                    )
                    st.success("✅ API Key is valid and working!")
                except Exception as e:
                    st.error(f"❌ API Key test failed: {str(e)}")
            else:
                st.warning("Please save an API key first")
    
    # API Key status
    if 'openai_api_key' in st.session_state:
        st.info("🔑 API Key is configured and ready for AI analysis")
    else:
        st.warning("⚠️ No API Key configured - AI Analysis features will be unavailable")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("CSV Data File")
        # CSV File uploader
        uploaded_csv = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to analyze",
            key="csv_uploader"
        )
        
        if uploaded_csv is not None:
            try:
                # Read the CSV
                df = pd.read_csv(uploaded_csv)
                
                # Store in session state
                st.session_state['data'] = df
                
                st.success(f"CSV file uploaded successfully! Shape: {df.shape}")
                
                # Show basic info
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Rows", df.shape[0])
                with col_b:
                    st.metric("Columns", df.shape[1])
                with col_c:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
        else:
            st.info("Please upload a CSV file")
    
    with col2:
        st.subheader("JSON Structure File")
        # JSON File uploader
        uploaded_json = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            help="Upload a JSON file with structured data",
            key="json_uploader"
        )
        
        if uploaded_json is not None:
            try:
                # Read the JSON
                json_data = json.load(uploaded_json)
                
                # Store in session state
                st.session_state['json_data'] = json_data
                
                st.success("JSON file uploaded successfully!")
                
                # Show basic info about JSON structure
                if isinstance(json_data, dict):
                    st.metric("Top-level keys", len(json_data.keys()))
                elif isinstance(json_data, list):
                    st.metric("Records", len(json_data))
                    if len(json_data) > 0 and isinstance(json_data[0], dict):
                        st.metric("Fields in first record", len(json_data[0].keys()))
                
            except Exception as e:
                st.error(f"Error reading JSON file: {str(e)}")
        else:
            st.info("Please upload a JSON file")
    
    with col3:
        st.subheader("Large Entity JSON File")
        # Large JSON File uploader
        uploaded_large_json = st.file_uploader(
            "Choose a large JSON file (Entity data)",
            type=['json'],
            help="Upload a large JSON file with company/entity organized data",
            key="large_json_uploader"
        )
        
        if uploaded_large_json is not None:
            try:
                # Read the large JSON
                large_json_data = json.load(uploaded_large_json)
                
                # Store in session state
                st.session_state['large_json_data'] = large_json_data
                
                st.success("Large JSON file uploaded successfully!")
                
                # Show basic info about large JSON structure
                if isinstance(large_json_data, dict):
                    st.metric("Companies/Entities", len(large_json_data.keys()))
                elif isinstance(large_json_data, list):
                    st.metric("Records", len(large_json_data))
                
                # Calculate approximate size
                import sys
                size_mb = sys.getsizeof(str(large_json_data)) / (1024 * 1024)
                st.metric("Approx Size", f"{size_mb:.1f} MB")
                
            except Exception as e:
                st.error(f"Error reading large JSON file: {str(e)}")
        else:
            st.info("Please upload a large JSON file")
    
    # Show combined status
    st.markdown("---")
    csv_status = "✅ CSV Loaded" if 'data' in st.session_state else "❌ CSV Not Loaded"
    json_status = "✅ JSON Loaded" if 'json_data' in st.session_state else "❌ JSON Not Loaded"
    large_json_status = "✅ Large JSON Loaded" if 'large_json_data' in st.session_state else "❌ Large JSON Not Loaded"
    st.write(f"**Status:** {csv_status} | {json_status} | {large_json_status}")
    
    # Preview sections
    if 'data' in st.session_state:
        st.subheader("CSV Data Preview")
        df = st.session_state['data']
        st.dataframe(df.head(), use_container_width=True)
        
        # Column information
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        with st.expander("Column Information"):
            st.dataframe(col_info, use_container_width=True)
    
    if 'json_data' in st.session_state:
        st.subheader("JSON Data Preview")
        json_data = st.session_state['json_data']
        
        # Show structure
        if isinstance(json_data, list) and len(json_data) > 0:
            st.write(f"JSON contains {len(json_data)} records")
            with st.expander("First Record Structure"):
                st.json(json_data[0])
        elif isinstance(json_data, dict):
            st.write("JSON structure:")
            with st.expander("JSON Structure Preview"):
                # Show just the keys and a sample to avoid overwhelming display
                preview = {}
                for key, value in list(json_data.items())[:3]:
                    if isinstance(value, dict):
                        preview[key] = {k: f"... ({type(v).__name__})" for k, v in list(value.items())[:3]}
                    elif isinstance(value, list):
                        preview[key] = f"[{len(value)} items]"
                    else:
                        preview[key] = value
                st.json(preview)
    
    if 'large_json_data' in st.session_state:
        st.subheader("Large JSON Data Preview")
        large_json_data = st.session_state['large_json_data']
        
        # Show structure for large JSON
        if isinstance(large_json_data, dict):
            st.write(f"Large JSON contains {len(large_json_data)} top-level entities/companies")
            with st.expander("Company/Entity Keys Sample"):
                # Show first few company names
                company_keys = list(large_json_data.keys())[:10]
                st.write("Sample companies:", company_keys)
                if len(large_json_data) > 10:
                    st.write(f"... and {len(large_json_data) - 10} more")
            
            with st.expander("First Company Structure"):
                first_company = list(large_json_data.keys())[0]
                first_company_data = large_json_data[first_company]
                if isinstance(first_company_data, dict):
                    preview = {}
                    for key, value in list(first_company_data.items())[:3]:
                        if isinstance(value, list):
                            preview[key] = f"[{len(value)} items]"
                        elif isinstance(value, dict):
                            preview[key] = f"{{dict with {len(value)} keys}}"
                        else:
                            preview[key] = value
                    st.json({first_company: preview})
        elif isinstance(large_json_data, list):
            st.write(f"Large JSON contains {len(large_json_data)} records")
            with st.expander("First Record Structure"):
                if len(large_json_data) > 0:
                    st.json(large_json_data[0])

elif page == "Data Analysis":
    st.header("Data Analysis")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        st.write(f"Analyzing CSV data with {df.shape[0]} rows and {df.shape[1]} columns")
        
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

elif page == "Individual Records":
    st.header("Individual Record Viewer")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Check if fullName column exists
        if 'fullName' in df.columns:
            # Get unique fullName values
            unique_names = sorted(df['fullName'].dropna().unique())
            
            if len(unique_names) > 0:
                # Select a person
                selected_name = st.selectbox(
                    "Select a person to view their details",
                    unique_names
                )
                
                if selected_name:
                    # Filter data for selected person
                    person_data = df[df['fullName'] == selected_name]
                    
                    st.subheader(f"Details for: {selected_name}")
                    st.write(f"Found {len(person_data)} record(s) for this person")
                    
                    # Convert to long form for each record
                    for idx, (record_idx, record) in enumerate(person_data.iterrows()):
                        if len(person_data) > 1:
                            st.write(f"**Record {idx + 1} (Row {record_idx + 1} in original data):**")
                        
                        # Create long form table
                        long_form_data = []
                        for col_name, value in record.items():
                            long_form_data.append({
                                'Field': col_name,
                                'Value': str(value) if pd.notna(value) else 'N/A'
                            })
                        
                        long_form_df = pd.DataFrame(long_form_data)
                        st.dataframe(long_form_df, use_container_width=True, hide_index=True)
                        
                        if len(person_data) > 1 and idx < len(person_data) - 1:
                            st.markdown("---")
            else:
                st.warning("No valid names found in the fullName column")
        else:
            st.error("Column 'fullName' not found in the uploaded data")
            st.write("Available columns:", ", ".join(df.columns.tolist()))
    else:
        st.warning("Please upload data first in the 'Data Upload' section")

elif page == "JSON Record Viewer":
    st.header("JSON Record Viewer")
    
    if 'json_data' in st.session_state:
        json_data = st.session_state['json_data']
        
        # Handle different JSON structures
        if isinstance(json_data, list):
            # JSON is a list of records
            full_names = []
            for record in json_data:
                if isinstance(record, dict) and 'fullName' in record:
                    full_names.append(record['fullName'])
            
            if len(full_names) > 0:
                unique_names = sorted(list(set(full_names)))
                
                selected_name = st.selectbox(
                    "Select a person to view their details",
                    unique_names
                )
                
                if selected_name:
                    # Find the record(s) for this person
                    person_records = [record for record in json_data if record.get('fullName') == selected_name]
                    
                    st.subheader(f"Details for: {selected_name}")
                    
                    for idx, record in enumerate(person_records):
                        if len(person_records) > 1:
                            st.write(f"**Record {idx + 1}:**")
                        
                        # Display hierarchical JSON structure with expanders
                        display_json_hierarchical(record)
                        
                        if len(person_records) > 1 and idx < len(person_records) - 1:
                            st.markdown("---")
            else:
                st.warning("No 'fullName' field found in JSON records")
        
        elif isinstance(json_data, dict):
            # JSON is a single object or structured data
            if 'fullName' in json_data:
                # Single record
                st.subheader(f"Details for: {json_data['fullName']}")
                display_json_hierarchical(json_data)
            else:
                # Check if it's a structure with nested records
                possible_records = []
                for key, value in json_data.items():
                    if isinstance(value, dict) and 'fullName' in value:
                        possible_records.append((key, value))
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and 'fullName' in item:
                                possible_records.append((f"{key} - {item.get('fullName')}", item))
                
                if possible_records:
                    record_options = [name for name, _ in possible_records]
                    selected_option = st.selectbox(
                        "Select a person to view their details",
                        record_options
                    )
                    
                    if selected_option:
                        selected_record = next(record for name, record in possible_records if name == selected_option)
                        st.subheader(f"Details for: {selected_record.get('fullName', 'Unknown')}")
                        display_json_hierarchical(selected_record)
                else:
                    st.warning("No 'fullName' field found in JSON structure")
                    st.write("JSON structure preview:")
                    st.json(json_data)
    else:
        st.warning("Please upload a JSON file first in the 'Data Upload' section")

elif page == "AI Analysis - Individual":
    st.header("AI Analysis - Individual")
    
    # Check if API key is available
    if 'openai_api_key' not in st.session_state:
        st.error("🔑 No OpenAI API Key configured!")
        st.info("Please go to the 'Data Upload' page and configure your OpenAI API key first.")
        st.stop()
    
    if 'json_data' in st.session_state:
        json_data = st.session_state['json_data']
        
        # Handle different JSON structures to find people
        people_options = []
        people_data = {}
        
        if isinstance(json_data, list):
            # JSON is a list of records
            for record in json_data:
                if isinstance(record, dict) and 'fullName' in record:
                    name = record['fullName']
                    people_options.append(name)
                    people_data[name] = record
        
        elif isinstance(json_data, dict):
            # JSON is a single object or structured data
            if 'fullName' in json_data:
                # Single record
                name = json_data['fullName']
                people_options.append(name)
                people_data[name] = json_data
            else:
                # Check if it's a structure with nested records
                for key, value in json_data.items():
                    if isinstance(value, dict) and 'fullName' in value:
                        name = value['fullName']
                        people_options.append(name)
                        people_data[name] = value
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and 'fullName' in item:
                                name = item['fullName']
                                people_options.append(name)
                                people_data[name] = item
        
        if people_options:
            unique_people = sorted(list(set(people_options)))
            
            selected_person = st.selectbox(
                "Select a person for AI analysis",
                unique_people
            )
            
            if selected_person:
                st.subheader(f"AI Analysis for: {selected_person}")
                
                # Show a preview of the data that will be analyzed
                with st.expander("Preview data to be analyzed", expanded=False):
                    st.json(people_data[selected_person])
                
                # Create two columns for the buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Extract Companies", use_container_width=True, key="btn_extract_companies"):
                        with st.spinner("Analyzing data with ChatGPT..."):
                            try:
                                # Set up OpenAI client using session state API key
                                client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
                                
                                # Prepare the data and query
                                person_json = json.dumps(people_data[selected_person], indent=2)
                                
                                query = """Review this JSON data and in the second column, I want please a full and complete read out of all the companies listed here. I want the list of companies, and I want a table of the first and second columns in full in which contain the companies. I don't want any of the rows which do not contain company names.
                                
Please format your response with:
1. A "List of Companies" section with bullet points
2. A "Table of Company Data" section with the relevant first and second columns

Here is the JSON data:

""" + person_json
                                
                                # Make API call to OpenAI
                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "user", "content": query}
                                    ],
                                    temperature=0.1
                                )
                                
                                # Store result in session state with different key
                                st.session_state['result_companies_analysis'] = response.choices[0].message.content
                                st.success("Analysis complete!")
                                
                            except Exception as e:
                                st.error(f"Error calling OpenAI API: {str(e)}")
                                st.info("Please check your API key and internet connection.")
                                # Debug info
                                st.write("OpenAI version:", openai.__version__ if hasattr(openai, '__version__') else "Unknown")
                
                with col2:
                    if st.button("Timeline Analysis", use_container_width=True, key="btn_timeline_analysis"):
                        with st.spinner("Analyzing timeline data with ChatGPT..."):
                            try:
                                # Set up OpenAI client using session state API key
                                client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
                                
                                # Prepare the data and query
                                person_json = json.dumps(people_data[selected_person], indent=2)
                                
                                query = """I wonder if you can at all figure out how to layer timeline over this. I'm interested in whatever timelines you can systematically extract which are either or both related to the career of this individual, or if we can extract any timelines etc around their movement in and around companies.

Please format your response with:
1. A "Career and Education Timeline" table with columns: Time Period, Company, Role/Title, Duration
2. An "Analysis of Career Movement and Timeline" section with detailed insights about career patterns, overlapping roles, progression, and any thematic consistency you can identify

Here is the JSON data:

""" + person_json
                                
                                # Make API call to OpenAI
                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "user", "content": query}
                                    ],
                                    temperature=0.1
                                )
                                
                                # Store result in session state with different key
                                st.session_state['result_timeline_analysis'] = response.choices[0].message.content
                                st.success("Timeline analysis complete!")
                                
                            except Exception as e:
                                st.error(f"Error calling OpenAI API: {str(e)}")
                                st.info("Please check your API key and internet connection.")
                                # Debug info
                                st.write("OpenAI version:", openai.__version__ if hasattr(openai, '__version__') else "Unknown")
                
                # Display results persistently
                if 'result_companies_analysis' in st.session_state:
                    st.markdown("---")
                    st.markdown("### Companies Analysis Results:")
                    st.markdown(st.session_state['result_companies_analysis'])
                
                if 'result_timeline_analysis' in st.session_state:
                    st.markdown("---")
                    st.markdown("### Timeline Analysis Results:")
                    st.markdown(st.session_state['result_timeline_analysis'])
                
                # Add styling for green buttons
                st.markdown("""
                <style>
                .stButton > button {
                    background-color: #28a745 !important;
                    border-color: #28a745 !important;
                    color: white !important;
                }
                .stButton > button:hover {
                    background-color: #218838 !important;
                    border-color: #1e7e34 !important;
                    color: white !important;
                }
                .stButton > button:focus {
                    background-color: #218838 !important;
                    border-color: #1e7e34 !important;
                    color: white !important;
                    box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.5) !important;
                }
                </style>
                """, unsafe_allow_html=True)
        
        else:
            st.warning("No records with 'fullName' field found in the JSON data")
    
    else:
        st.warning("Please upload a JSON file first in the 'Data Upload' section")

elif page == "AI Analysis - Entity":
    st.header("AI Analysis - Entity")
    
    # Check if API key is available
    if 'openai_api_key' not in st.session_state:
        st.error("🔑 No OpenAI API Key configured!")
        st.info("Please go to the 'Data Upload' page and configure your OpenAI API key first.")
        st.stop()
    
    if 'large_json_data' in st.session_state:
        large_json_data = st.session_state['large_json_data']
        
        # Debug section to understand JSON structure
        st.subheader("JSON Structure Analysis")
        
        with st.expander("Debug: Analyze JSON Structure", expanded=True):
            st.write("**Data Type:**", type(large_json_data).__name__)
            
            if isinstance(large_json_data, dict):
                st.write("**Number of top-level keys:**", len(large_json_data))
                st.write("**First 10 top-level keys:**")
                keys_sample = list(large_json_data.keys())[:10]
                for i, key in enumerate(keys_sample):
                    st.write(f"{i+1}. `{key}` (type: {type(large_json_data[key]).__name__})")
                
                # Show structure of first key
                if len(large_json_data) > 0:
                    first_key = list(large_json_data.keys())[0]
                    first_value = large_json_data[first_key]
                    st.write(f"**Structure of first key '{first_key}':**")
                    
                    if isinstance(first_value, dict):
                        st.write("- It's a dictionary with keys:", list(first_value.keys())[:10])
                    elif isinstance(first_value, list):
                        st.write(f"- It's a list with {len(first_value)} items")
                        if len(first_value) > 0:
                            st.write(f"- First item type: {type(first_value[0]).__name__}")
                            if isinstance(first_value[0], dict):
                                st.write(f"- First item keys: {list(first_value[0].keys())[:10]}")
                    else:
                        st.write(f"- It's a {type(first_value).__name__}: {str(first_value)[:100]}...")
            
            elif isinstance(large_json_data, list):
                st.write("**Number of items in list:**", len(large_json_data))
                if len(large_json_data) > 0:
                    st.write("**First item type:**", type(large_json_data[0]).__name__)
                    if isinstance(large_json_data[0], dict):
                        st.write("**First item keys:**", list(large_json_data[0].keys())[:10])
        
        # Try different approaches to extract companies
        st.subheader("Company Extraction")
        
        company_options = []
        company_data_map = {}
        extraction_method = ""
        
        if isinstance(large_json_data, list):
            # Method 1: List of company objects with "company_name" field
            st.write("**Trying Method 1: List of company objects with 'company_name' field**")
            
            for item in large_json_data:
                if isinstance(item, dict) and 'company_name' in item:
                    company_name = item['company_name']
                    company_options.append(company_name)
                    company_data_map[company_name] = item
            
            if company_options:
                extraction_method = "company_name_field"
                st.success(f"✅ Found {len(company_options)} companies using 'company_name' field")
            else:
                st.warning("No 'company_name' field found in list items")
        
        elif isinstance(large_json_data, dict):
            # Method 2: Top-level keys are companies
            if len(large_json_data) > 0:
                st.write("**Trying Method 2: Top-level keys as companies**")
                sample_keys = list(large_json_data.keys())[:5]
                st.write("Sample keys:", sample_keys)
                
                # Use top-level keys as companies
                for company_name, company_data in large_json_data.items():
                    company_options.append(company_name)
                    company_data_map[company_name] = company_data
                
                extraction_method = "top_level_keys"
        
        if company_options:
            st.success(f"Found {len(company_options)} companies using method: {extraction_method}")
            
            # Sort companies alphabetically
            sorted_companies = sorted(company_options)
            
            selected_company = st.selectbox(
                "Select a company/entity for AI analysis",
                sorted_companies,
                help=f"Choose from {len(sorted_companies)} available companies"
            )
            
            if selected_company:
                st.subheader(f"AI Analysis for Company: {selected_company}")
                
                # Get the company data
                company_data = company_data_map.get(selected_company, {})
                
                # Show info about the company data
                if company_data:
                    st.write(f"**Data type for {selected_company}:** {type(company_data).__name__}")
                    
                    if isinstance(company_data, dict):
                        st.write(f"**Number of fields:** {len(company_data)}")
                        st.write(f"**Field names:** {list(company_data.keys())}")
                        
                        # Count individuals in associated_persons if it exists
                        if 'associated_persons' in company_data:
                            associated_persons = company_data['associated_persons']
                            if isinstance(associated_persons, list):
                                st.write(f"**Number of associated persons:** {len(associated_persons)}")
                            elif isinstance(associated_persons, dict):
                                st.write(f"**Associated persons structure:** {len(associated_persons)} top-level keys")
                        
                        # Show association_map info if it exists
                        if 'association_map' in company_data:
                            association_map = company_data['association_map']
                            if isinstance(association_map, dict):
                                st.write(f"**Association map:** {len(association_map)} entries")
                            elif isinstance(association_map, list):
                                st.write(f"**Association map:** {len(association_map)} items")
                
                # Show a preview of the company data that will be analyzed
                with st.expander("Preview company data to be analyzed", expanded=False):
                    if company_data:
                        # Show a limited preview to avoid overwhelming the display
                        preview_data = {}
                        for key, value in company_data.items():
                            if isinstance(value, list):
                                preview_data[key] = f"[{len(value)} items]" if len(value) > 10 else value[:2]
                            elif isinstance(value, dict):
                                preview_data[key] = f"{{dict with {len(value)} keys}}"
                            else:
                                preview_data[key] = value
                        st.json(preview_data)
                    else:
                        st.warning("No data found for this company")
                
                # Create two columns for the analysis buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    # Intelligence Analysis button
                    if st.button("Intelligence Analysis", use_container_width=True, key="btn_intelligence_analysis"):
                        if company_data:
                            with st.spinner(f"Conducting intelligence analysis for {selected_company}..."):
                                try:
                                    # Set up OpenAI client using session state API key
                                    client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
                                    
                                    # Prepare the data and query
                                    company_json = json.dumps(company_data, indent=2)
                                    
                                    # Check if the JSON is too large for the API (more conservative limit)
                                    if len(company_json) > 300000:  # ~300KB limit to stay under token limits
                                        st.warning("Company data is large. Creating optimized summary for analysis.")
                                        # Create a more detailed summary for large datasets
                                        summary_data = {
                                            "company_name": company_data.get("company_name", selected_company),
                                            "total_associated_persons": 0,
                                            "association_summary": {},
                                            "sample_profiles": [],
                                            "geographic_distribution": {},
                                            "role_distribution": {},
                                            "skills_summary": {}
                                        }
                                        
                                        # Process associated_persons if it exists
                                        if 'associated_persons' in company_data and isinstance(company_data['associated_persons'], list):
                                            associated_persons = company_data['associated_persons']
                                            summary_data["total_associated_persons"] = len(associated_persons)
                                            
                                            # Sample up to 10 detailed profiles
                                            summary_data["sample_profiles"] = associated_persons[:10]
                                            
                                            # Analyze locations, roles, and skills
                                            locations = {}
                                            roles = {}
                                            skills = set()
                                            
                                            for person in associated_persons[:50]:  # Analyze first 50 for patterns
                                                # Extract location info
                                                for key in ['location', 'city', 'country', 'address']:
                                                    if isinstance(person, dict) and key in person:
                                                        loc = person[key]
                                                        if loc and isinstance(loc, str):
                                                            locations[loc] = locations.get(loc, 0) + 1
                                                
                                                # Extract role info
                                                for key in ['title', 'role', 'position', 'job_title']:
                                                    if isinstance(person, dict) and key in person:
                                                        role = person[key]
                                                        if role and isinstance(role, str):
                                                            roles[role] = roles.get(role, 0) + 1
                                                
                                                # Extract skills safely
                                                if isinstance(person, dict) and 'skills' in person:
                                                    person_skills = person['skills']
                                                    if isinstance(person_skills, list):
                                                        for skill in person_skills[:5]:
                                                            if isinstance(skill, str):
                                                                skills.add(skill)
                                                            elif isinstance(skill, dict) and 'name' in skill:
                                                                skills.add(skill['name'])
                                                    elif isinstance(person_skills, str):
                                                        skills.add(person_skills)
                                            
                                            summary_data["geographic_distribution"] = dict(list(locations.items())[:20])
                                            summary_data["role_distribution"] = dict(list(roles.items())[:20])
                                            summary_data["skills_summary"] = list(skills)[:50]
                                        
                                        # Add association_map summary if it exists
                                        if 'association_map' in company_data:
                                            association_map = company_data['association_map']
                                            if isinstance(association_map, dict):
                                                summary_data["association_summary"] = {
                                                    "total_mappings": len(association_map),
                                                    "sample_mappings": dict(list(association_map.items())[:10])
                                                }
                                            elif isinstance(association_map, list):
                                                summary_data["association_summary"] = {
                                                    "total_mappings": len(association_map),
                                                    "sample_mappings": association_map[:10]
                                                }
                                        
                                        company_json = json.dumps(summary_data, indent=2)
                                        st.info(f"Sending optimized summary: {len(company_json)} characters")
                                    else:
                                        st.info(f"Sending full company data: {len(company_json)} characters")
                                    
                                    # Intelligence analysis query
                                    query = f"""Conduct an intelligence analysis of {selected_company} based on the personnel and relationship data provided. Approach this as a strategic intelligence assessment focusing on:

**ORGANIZATIONAL INTELLIGENCE:**
1. **Corporate Footprint**: What does the composition of associated personnel reveal about the company's true scope, capabilities, and market positioning?
2. **Strategic Assets**: Identify key individuals who represent critical capabilities, connections, or competitive advantages
3. **Operational Patterns**: What patterns in roles, backgrounds, and relationships indicate the company's operational model and business priorities?

**NETWORK & INFLUENCE ANALYSIS:**
4. **Power Structure**: Map the apparent hierarchy and influence networks - who are the key decision makers and connectors?
5. **External Reach**: What external organizations, institutions, or sectors does this company connect to through its people?
6. **Geographic Presence**: Analyze geographic distribution to understand operational territories and regional influence

**CAPABILITY ASSESSMENT:**
7. **Core Competencies**: Based on personnel backgrounds, what are the company's demonstrated capabilities and technical expertise?
8. **Innovation Capacity**: Evidence of R&D, technical leadership, or cutting-edge expertise within the organization
9. **Market Position**: How do the backgrounds and connections suggest this company's competitive position and market influence?

**RISK & OPPORTUNITY FACTORS:**
10. **Dependencies**: Key person dependencies or single points of failure in expertise/relationships
11. **Growth Indicators**: Signs of expansion, new capabilities, or strategic shifts based on recent hires or role changes
12. **Competitive Intelligence**: What does this reveal about the company's strategy, partnerships, or market approach?

Provide a comprehensive intelligence briefing with specific evidence from the data to support your assessments.

Here is the company data for analysis:

""" + company_json
                                    
                                    # Make API call to OpenAI
                                    response = client.chat.completions.create(
                                        model="gpt-4o",
                                        messages=[
                                            {"role": "user", "content": query}
                                        ],
                                        temperature=0.1
                                    )
                                    
                                    # Store result in session state
                                    st.session_state['result_intelligence_analysis'] = response.choices[0].message.content
                                    st.success("Intelligence analysis complete!")
                                    
                                except Exception as e:
                                    st.error(f"Error calling OpenAI API: {str(e)}")
                                    st.info("Please check your API key and internet connection.")
                                    if "token" in str(e).lower():
                                        st.info("The company data might be too large. Try selecting a smaller company or we can implement data chunking.")
                        else:
                            st.error("No data available for the selected company")
                
                with col2:
                    # Structured Table Analysis button
                    if st.button("Structured Table Analysis", use_container_width=True, key="btn_structured_analysis"):
                        if company_data:
                            with st.spinner(f"Creating structured analysis for {selected_company}..."):
                                try:
                                    # Set up OpenAI client using session state API key
                                    client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
                                    
                                    # Prepare the data and query
                                    company_json = json.dumps(company_data, indent=2)
                                    
                                    # Check if the JSON is too large for the API (more conservative limit)
                                    if len(company_json) > 300000:  # ~300KB limit to stay under token limits
                                        st.warning("Company data is large. Creating optimized summary for structured analysis.")
                                        # Create a more detailed summary for large datasets
                                        summary_data = {
                                            "company_name": company_data.get("company_name", selected_company),
                                            "total_associated_persons": 0,
                                            "association_summary": {},
                                            "detailed_profiles": [],
                                            "geographic_distribution": {},
                                            "role_distribution": {},
                                            "skills_summary": {},
                                            "timeline_data": {}
                                        }
                                        
                                        # Process associated_persons if it exists
                                        if 'associated_persons' in company_data and isinstance(company_data['associated_persons'], list):
                                            associated_persons = company_data['associated_persons']
                                            summary_data["total_associated_persons"] = len(associated_persons)
                                            
                                            # Sample up to 15 detailed profiles for table analysis
                                            summary_data["detailed_profiles"] = associated_persons[:15]
                                            
                                            # Analyze locations, roles, skills, and timeline data
                                            locations = {}
                                            roles = {}
                                            skills = set()
                                            timeline_info = {}
                                            
                                            for person in associated_persons[:75]:  # Analyze more for table creation
                                                # Extract location info
                                                for key in ['location', 'city', 'country', 'address']:
                                                    if isinstance(person, dict) and key in person:
                                                        loc = person[key]
                                                        if loc and isinstance(loc, str):
                                                            locations[loc] = locations.get(loc, 0) + 1
                                                
                                                # Extract role info
                                                for key in ['title', 'role', 'position', 'job_title']:
                                                    if isinstance(person, dict) and key in person:
                                                        role = person[key]
                                                        if role and isinstance(role, str):
                                                            roles[role] = roles.get(role, 0) + 1
                                                
                                                # Extract skills safely
                                                if isinstance(person, dict) and 'skills' in person:
                                                    person_skills = person['skills']
                                                    if isinstance(person_skills, list):
                                                        for skill in person_skills[:10]:
                                                            if isinstance(skill, str):
                                                                skills.add(skill)
                                                            elif isinstance(skill, dict) and 'name' in skill:
                                                                skills.add(skill['name'])
                                                    elif isinstance(person_skills, str):
                                                        skills.add(person_skills)
                                                
                                                # Extract timeline/date information safely
                                                for key in ['start_date', 'end_date', 'join_date', 'years', 'duration']:
                                                    if isinstance(person, dict) and key in person:
                                                        value = person[key]
                                                        if value and isinstance(value, (str, int, float)):
                                                            timeline_info[key] = timeline_info.get(key, [])
                                                            timeline_info[key].append(str(value))
                                            
                                            summary_data["geographic_distribution"] = dict(list(locations.items())[:30])
                                            summary_data["role_distribution"] = dict(list(roles.items())[:30])
                                            summary_data["skills_summary"] = list(skills)[:100]
                                            summary_data["timeline_data"] = timeline_info
                                        
                                        # Add association_map summary if it exists
                                        if 'association_map' in company_data:
                                            association_map = company_data['association_map']
                                            if isinstance(association_map, dict):
                                                summary_data["association_summary"] = {
                                                    "total_mappings": len(association_map),
                                                    "sample_mappings": dict(list(association_map.items())[:15])
                                                }
                                            elif isinstance(association_map, list):
                                                summary_data["association_summary"] = {
                                                    "total_mappings": len(association_map),
                                                    "sample_mappings": association_map[:15]
                                                }
                                        
                                        company_json = json.dumps(summary_data, indent=2)
                                        st.info(f"Sending optimized summary for tables: {len(company_json)} characters")
                                    else:
                                        st.info(f"Sending full company data: {len(company_json)} characters")
                                    
                                    # Structured table analysis query
                                    query = f"""Create a comprehensive structured table analysis of {selected_company} showing how the company presents across all associated people. Format as detailed tables with the following information using ACTUAL DATA from the JSON (no interpretations or assessments):

**INSTRUCTIONS:**
- Create structured tables (use markdown table format)
- Use ONLY actual data fields from the JSON - no analytical assessments
- Include timelines where possible from the data
- Show the connection type/field that links each person to the company
- Extract raw data values, not summaries

**TABLE 1: PERSONNEL OVERVIEW**
| Full Name | Primary Role/Title | Connection Type/Field | Start Date | End Date | Location | Department |

**TABLE 2: GEOGRAPHIC DISTRIBUTION**
| Location/Region | Person Names | Their Actual Roles | Time Period | Contact Info |

**TABLE 3: ORGANIZATIONAL STRUCTURE**
| Department/Function | Person Names | Their Actual Titles | Actual Start Dates | Reporting Structure |

**TABLE 4: TEMPORAL ANALYSIS**
| Time Period | Person Name | Action (Join/Leave/Role Change) | From Role | To Role | Actual Dates |

**TABLE 5: CONNECTION MAPPING**
| Person Name | Connection Field/Type | Data Source | Contact Details | Social Profiles | Other Companies |

**TABLE 6: CAPABILITY MATRIX**
| Skill/Expertise Area | Person Names | Actual Experience Years | Specific Projects | Certifications | Education |

**CRITICAL REQUIREMENTS:**
- Extract ONLY actual field values from the JSON data
- Do NOT create analytical assessments like "High", "Medium", "Strong" 
- Show real names, real dates, real titles, real locations from the data
- If a field doesn't exist in the data, show "N/A"
- Include actual URLs, email addresses, phone numbers if present
- Show specific project names, company names, degree titles, certification names

Analyze the provided data and create these structured tables with ONLY factual information extracted directly from the company data.

Here is the company data for analysis:

""" + company_json
                                    
                                    # Make API call to OpenAI
                                    response = client.chat.completions.create(
                                        model="gpt-4o",
                                        messages=[
                                            {"role": "user", "content": query}
                                        ],
                                        temperature=0.1
                                    )
                                    
                                    # Store result in session state
                                    st.session_state['result_structured_analysis'] = response.choices[0].message.content
                                    st.success("Structured analysis complete!")
                                    
                                except Exception as e:
                                    st.error(f"Error calling OpenAI API: {str(e)}")
                                    st.info("Please check your API key and internet connection.")
                                    if "token" in str(e).lower():
                                        st.info("The company data might be too large. Try selecting a smaller company or we can implement data chunking.")
                        else:
                            st.error("No data available for the selected company")
                
                # Display results persistently
                if 'result_intelligence_analysis' in st.session_state:
                    st.markdown("---")
                    st.markdown("### Intelligence Analysis Results:")
                    st.markdown(st.session_state['result_intelligence_analysis'])
                
                if 'result_structured_analysis' in st.session_state:
                    st.markdown("---")
                    st.markdown("### Structured Table Analysis Results:")
                    st.markdown(st.session_state['result_structured_analysis'])
                
                # Add styling for green buttons
                st.markdown("""
                <style>
                .stButton > button {
                    background-color: #28a745 !important;
                    border-color: #28a745 !important;
                    color: white !important;
                }
                .stButton > button:hover {
                    background-color: #218838 !important;
                    border-color: #1e7e34 !important;
                    color: white !important;
                }
                .stButton > button:focus {
                    background-color: #218838 !important;
                    border-color: #1e7e34 !important;
                    color: white !important;
                    box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.5) !important;
                }
                </style>
                """, unsafe_allow_html=True)
        
        else:
            st.warning("No companies/entities found in the large JSON data")
            st.info("The JSON structure might be different than expected. Please check the debug information above.")
    
    else:
        st.warning("Please upload a large JSON file first in the 'Data Upload' section")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Privacy Note**: All data processing happens locally on your machine. No data is sent to external servers.")