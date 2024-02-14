import pandas as pd
import streamlit as st

def handle_missing_values(df):
    st.write("Let's handle missing values or duplicates.")
    column_names = None
    method = st.selectbox(
        "Select Method",
        ["Drop Missing Values", "Fill Missing Values", "Show Duplicates"]
    )

    if method != "Show Duplicates":
        column_names = st.multiselect("Select Columns", df.columns)

    if method == "Drop Missing Values":
        df_cleaned = df.dropna(subset=column_names)
    elif method == "Fill Missing Values":
        fill_method_options = ["Fill with Zero", "Fill with Mean", "Fill with Median", "Custom Value"]
        fill_methods = {}

        for col in column_names:
            fill_methods[col] = st.selectbox(f"Select Fill Method for '{col}'", fill_method_options)

        df_cleaned = df.copy()
        for col, fill_method in fill_methods.items():
            if fill_method == "Fill with Zero":
                df_cleaned[col] = df_cleaned[col].fillna(0)
            elif fill_method == "Fill with Mean":
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif fill_method == "Fill with Median":
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            else:
                custom_value = st.number_input(f"Enter Custom Value for '{col}'")
                df_cleaned[col] = df_cleaned[col].fillna(custom_value)
    else:  # For "Show Duplicates"
        duplicate_count = df.duplicated(subset=column_names).sum()
        st.write(f"Number of duplicate rows: {duplicate_count}")
        df_cleaned = df  # No modification for this method

    with st.expander("Dataframe Preview After Preprocessing"):
        st.write(df_cleaned)
    
    st.session_state['df'] = df_cleaned
    
    # Show count of missing values for each column in a bar chart
    missing_values_count = df_cleaned.isnull().sum()
    st.bar_chart(missing_values_count)
    






st.title('Basic Preprocessing Of Dataset')
st.write("Welcome to Our Chatbot!")
st.write("In this section, you will handle missing values and duplicate rows.")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    with st.expander("Dataframe Preview"):
        st.write(df)
    handle_missing_values(df)
