import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def handle_duplicates(df):
    st.write("### Duplicate Handling")
    duplicate_count = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicate_count}")

    delete_duplicates = st.selectbox("Do you want to handle the duplicates?", ("No", "Yes"))

    if delete_duplicates == "Yes" and duplicate_count > 0:
        df_cleaned = df.drop_duplicates()
        st.write("Duplicates removed.")
    else:
        df_cleaned = df  # No modification

    return df_cleaned, delete_duplicates

def drop_columns(df):
    st.write("### Drop Columns")
    column_names = df.columns
    selected_columns_to_drop = st.multiselect("Select Columns to Drop", column_names)

    if selected_columns_to_drop:
        df_cleaned = df.drop(columns=selected_columns_to_drop)
        st.write("Selected columns dropped.")
    else:
        df_cleaned = df
        st.warning("Please select columns to drop.")

    return df_cleaned

def handle_missing_values(df):
    st.write("### Missing Values Count")
    missing_values_count = df.isnull().sum()
    st.bar_chart(missing_values_count)

    st.write("### Missing Value Handling")
    column_names = df.columns
    selected_columns = st.multiselect("Select Columns to Handle Missing Values", column_names)

    if selected_columns:
        handling_options = ["Drop Missing Values", "Fill Missing Values", "Finish"]
        method = st.selectbox("Select Method", handling_options)

        selected_handling_methods = {}
        for col in selected_columns:
            selected_handling_methods[col] = st.selectbox(f"Select Handling Method for '{col}'", handling_options[:-1])

        df_cleaned = df.copy()
        for col, method in selected_handling_methods.items():
            if method == "Drop Missing Values":
                df_cleaned = df_cleaned.dropna(subset=[col])
            elif method == "Fill Missing Values":
                fill_method_options = ["Fill with Zero", "Fill with Mean", "Fill with Median", "Custom Value"]
                fill_method = st.selectbox(f"Select Fill Method for '{col}'", fill_method_options)
                if fill_method == "Fill with Zero":
                    df_cleaned[col] = df_cleaned[col].fillna(0)
                elif fill_method == "Fill with Mean":
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                elif fill_method == "Fill with Median":
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                else:
                    custom_value = st.number_input(f"Enter Custom Value for '{col}'")
                    df_cleaned[col] = df_cleaned[col].fillna(custom_value)

    else:
        df_cleaned = df
        st.warning("Please select columns to handle missing values.")

    return df_cleaned

def handle_encoding(df):
    st.write("### Encoding")
    column_names = df.columns
    selected_columns_to_encode = st.multiselect("Select Columns for Encoding", column_names)

    if selected_columns_to_encode:
        df_encoded = df.copy()

        for col in selected_columns_to_encode:
            st.write(f"#### Encoding for column: {col}")
            encoding_option = st.selectbox(f"Select Encoding Method for '{col}'", ("Label Encoding", "One-Hot Encoding"))

            if encoding_option == "Label Encoding":
                label_encoder = LabelEncoder()
                df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
                st.write(f"Label Encoding applied for '{col}'.")
            elif encoding_option == "One-Hot Encoding":
                encoded_col = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, encoded_col], axis=1)
                df_encoded.drop(columns=[col], inplace=True)
                st.write(f"One-Hot Encoding applied for '{col}'.")

        return df_encoded

    else:
        st.warning("Please select columns for encoding.")
        return df
    
    
def handle_scaling(df):
    st.write("### Scaling")
    column_names = df.columns
    selected_columns_to_scale = st.multiselect("Select Columns for Scaling", column_names)

    if selected_columns_to_scale:
        scaler = StandardScaler()
        scaled_columns = scaler.fit_transform(df[selected_columns_to_scale])
        df_scaled = df.copy()
        df_scaled[selected_columns_to_scale] = scaled_columns
        st.write("Standard Scaling applied.")
        return df_scaled

    else:
        st.warning("Please select columns for scaling.")
        return df
    
    
def plot_graph(df, x_col, y_col, plot_type):
    plt.figure(figsize=(8, 6))
    
    if plot_type == "Bar Plot":
        df.groupby(x_col)[y_col].mean().plot(kind='bar')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Bar Plot: {y_col} vs {x_col}')
        st.pyplot()
    elif plot_type == "Scatter Plot":
        plt.scatter(df[x_col], df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Scatter Plot: {y_col} vs {x_col}')
        st.pyplot()
    
    

st.title('🖐️ Hi Lets chat with Your data')
st.write("Welcome to Our Chatbot!")
st.write("This Chatbot will do some data Preprocessing steps and also shows you the Visualization at the end ")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.info("CSV Uploaded Successfully")
    with st.sidebar.expander("Dataframe Preview"):
        st.write(df)

    df, delete_duplicates = handle_duplicates(df)

    if delete_duplicates == "Yes":
        df = drop_columns(df)

        show_missing_values = st.radio("Show Missing Values", ("Yes", "No"), index=1)
        if show_missing_values == "Yes":
            df = handle_missing_values(df)
            
        show_encoding = st.radio("Perform Encoding", ("Yes", "No"), index=1)
        if show_encoding == "Yes":
            df = handle_encoding(df)
            
        show_scaling = st.radio("Perform Scaling", ("Yes", "No"), index=1)
        if show_scaling == "Yes":
            df = handle_scaling(df)
            
        if st.button("Show Final Processed DataFrame"):
            with st.expander("View Final DataFrame"):
                st.write(df)
                
        show_visualization = st.radio("Performing Visualization", ("Yes", "No"), index=1)
        if show_visualization == "Yes":
            st.write("Select two columns for visualization:")
            columns_for_plot = st.multiselect("Select Columns", df.columns.tolist(), key='visualization')

            if len(columns_for_plot) == 2:
                plot_option = st.selectbox("Select Plot Type", ("Bar Plot", "Scatter Plot"))

            if st.button("Generate Plot"):
                plot_col1, plot_col2 = columns_for_plot
                plot_graph(df, plot_col1, plot_col2, plot_option)
