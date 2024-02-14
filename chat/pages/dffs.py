import streamlit as st
import pandas as pd
from llama_index.query_engine import PandasQueryEngine
from IPython.display import Markdown , display
from llama_index.llms.palm import PaLM
import os
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex
from llama_index import (
    ServiceContext,
    LLMPredictor,
    # PromptHelper,
)

from llama_index import download_loader
from pathlib import Path

os.environ["OPENAI_API_KEY"] = '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'
# os.environ['GOOGLE_API_KEY'] = 'AIzaSyDkU4f8ZGCSk8klzgaplc-ODrYOc6wmQCU'
df = st.session_state['df'] # updated dataframe

PandasCSVReader = download_loader("PandasCSVReader")
loader = PandasCSVReader()
documents = loader.load_data(file=Path('train.csv'))
# documents = df.apply(lambda row: {'id': row['PassengerId'], 'text': row['Survived'],'text': row['Pclass'],'text': row['Name'],'text': row['Sex'],'text': row['Age'],'text': row['Age'],'text': row['Age'],'text': row['SibSp'],'text': row['Parch'],'text': row['Ticket'],'text': row['Fare'],'text': row['Cabin'],'text': row['Embarked']}, axis=1).tolist()
# documents = df.apply(lambda row: {'id': row['PassengerId'], 'text': ' '.join(str(value) for value in row)}, axis=1).tolist()
llm_predictor = LLMPredictor(llm = PaLM()) 


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2") 

# Create a ServiceContext with the LLM
service_context = ServiceContext.from_defaults(
    # llm_predictor=llm_predictor,
    embed_model=embed_model
    )

# Create the index with the new embedding model
index = VectorStoreIndex.from_documents(
    documents=documents, 
    service_context=service_context,
)





st.title("😊 Lets Start Chat with Dataset")
st.write("Let's handle missing values or duplicates.")

    


col1, col2 = st.columns([1,1])

with col1:
    st.info("CSV Uploaded Successfully")
     
    with st.expander("Dataframe Preview"):
        st.write(df)

with col2:

    st.info("Chat Below")
    
    query = st.text_area("Chat with Dataframe")
    st.write(query)
    
     
    if query:
        # Create the query engine
        query_engine = PandasQueryEngine(df=df, index=index, verbose=True)
        response = query_engine.query(query)
        st.write(response)
        
        




# # Specify your new LLM and embedding model
# new_llm = "your_new_llm"  # replace with the name of your new LLM
# new_embedding_model = "your_new_embedding_model"  # replace with the name of your new embedding model

# # Create the index with the new embedding model
# index = VectorStoreIndex.from_documents(
#     documents=df, 
#     service_context=service_context, 
#     embedding_model=new_embedding_model
# )

# # Create the query engine
# query_engine = PandasQueryEngine(df=df, index=index, verbose=True)

# # Update the LLM
# query_engine.llm = new_llm