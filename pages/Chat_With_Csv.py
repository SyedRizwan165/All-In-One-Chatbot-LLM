# Import necessary libraries
import streamlit as st 
from pandasai.llm.openai import OpenAI
import os
import pandas as pd
from pandasai import PandasAI
from PIL import Image
from pandasai import SmartDataframe

# Setup your OPENAI key
OPENAI_API_KEY = ""

# Function to chat with CSV using OpenAI
def csv_file(df, prompt):
    # Initialize OpenAI language model
    llm = OpenAI(api_token=OPENAI_API_KEY)
    # Initialize Pandas AI for DataFrame processing
    pandas_ai = PandasAI(llm)
    # Run the chat with the DataFrame and prompt
    result = pandas_ai.run(df, prompt=prompt)
    # Print the result 
    print(result)
    return result

# Set up Streamlit app layout
st.set_page_config(layout='wide')
st.title("Chat With Your CSV 📊")

# File upload widget for CSV
input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv is not None:
    st.info("CSV Uploaded Successfully")
    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_csv)
    # Wrap the DataFrame in SmartDataframe for enhanced functionality
    df = SmartDataframe(data)
    # Display the DataFrame in the app
    answer = df.dataframe
    st.dataframe(answer)
    
    # Text area for user to enter a query
    input_query = st.text_area("Enter your query")
    
    if input_query is not None:
      
    # Button to trigger the generation of the response
     if st.button("Generate"):
        st.info("Your Query: " + input_query)
        
        # Call the chat_with_csv function to generate a response
        result = csv_file(data, input_query)
        
        # Check if keywords for visualization are present in the query
        visualization_keywords = ['create', 'draw', 'plot', 'graph']
        generate_visualization = any(keyword in input_query.lower() for keyword in visualization_keywords)
        
        if generate_visualization:
            # Generate and display the visualization
            image = Image.open('temp_chart.png') 
            st.image(image)
        else:
            # Display the response in text form
            st.success(result)

        

                    
