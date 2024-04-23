import streamlit as st
import re
import pymysql
import pandas as pd
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
 
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
 
 
def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
 
 
 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
 
 
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
 
 
def get_conversational_chain():
 
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,don't provide the wrong answer\n\n
    If the context does not mention any of the requested details, take 'null' for data.
    Context:\n {context}?\n
    Question: \n{question}\n
 
    Answer:
    """
 
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
 
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
 
    return chain
 
 
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
 
    chain = get_conversational_chain()
 
   
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
 
    return response["output_text"]
 
def answer_question(question, text):
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(text)
    get_vector_store(text_chunks)
 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(text)  # Use the text itself as context
 
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )
    return response["output_text"]
 
def create_docs(user_pdf_list):
 
    df = pd.DataFrame({'Invoice #': pd.Series(dtype='str'),
                       'DESCRIPTION': pd.Series(dtype='str'),
                       'Invoice Date': pd.Series(dtype='str'),
                       'Due Date': pd.Series(dtype='str'),
                       'AMOUNT': pd.Series(dtype='str'),
                       'TOTAL': pd.Series(dtype='str'),
                       'Contact email': pd.Series(dtype='str'),
                       'User email': pd.Series(dtype='str'),
                       'Contact number': pd.Series(dtype='str'),
                       'Bill To': pd.Series(dtype='str')
                        })
 
    for filename in user_pdf_list:
 
        print(filename)
        raw_text = get_pdf_text(filename)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
 
        question = """Extract all the following values : Invoice # (or) INVOICE NO, DESCRIPTION, Invoice Date, Due Date, AMOUNT, TOTAL, Contact email, Contact number (or) phone number and Bill To (or) address from this content.
        for the differnet DESCRIPTION create different dictionaries structure of all data and put all that in the single list structure and if any data is missing  take 'null'for data.
       
 
        Expected output format: remove any dollar symbols, also for '\n' use ',' [{'Invoice #': '1001329','DESCRIPTION': 'Office Chair','Invoice Date': '5/4/2023',Due Date: '10/4/2023','AMOUNT': '2000.00','TOTAL': '2200.00','Contact email': 'Santoshvarma0988@gmail.com','User email':'bsktrending@gmail.com',Contact number': '9999999999','Bill To': 'Mumbai, India'}, {'Invoice #': '1001329','DESCRIPTION': 'Pen','Invoice Date': '5/4/2023','AMOUNT': '200.00','TOTAL': '2200.00','Contact email': 'Santoshvarma0988@gmail.com','Contact number': '9999999999','Bill To': 'Mumbai, India'}]
        """
       
        print('Extract data using Hugging Face Transformers')
        llm_extracted_data = str(user_input(question))
 
       
        pattern = r"{([^}]+)}"  # Matches curly braces with content inside
        matches = re.findall(pattern, llm_extracted_data)
 
        for substring in matches:
            substring = '{'+substring+'}'
            pattern2 = r'{(.+)}'
            match = re.search(pattern2, substring, re.DOTALL)
            if match:
                extracted_text = match.group(1)
                print('Converting the extracted text to a dictionary')
                data_dict = eval('{' + extracted_text + '}')
                # Convert Invoice Date format from 'DD/MM/YYYY' to 'YYYY-MM-DD'
                try:
                    invoice_date = datetime.strptime(data_dict['Invoice Date'], '%d/%m/%Y').strftime('%Y-%m-%d')
                    data_dict['Invoice Date'] = invoice_date
                except ValueError:
                    data_dict['Invoice Date'] = 'null'

                # Convert Due Date format from 'DD/MM/YYYY' to 'YYYY-MM-DD'
                try:
                    due_date = datetime.strptime(data_dict['Due Date'], '%d/%m/%Y').strftime('%Y-%m-%d')
                    data_dict['Due Date'] = due_date
                except ValueError:
                    data_dict['Due Date'] = 'null'

                # Remove commas and currency symbol from 'TOTAL' column values
                if 'TOTAL' in data_dict:
                    total = data_dict['TOTAL'].replace(',', '').replace('₹', '')
                    data_dict['TOTAL'] = total
                # Remove commas from 'AMOUNT' column values
                if 'AMOUNT' in data_dict:
                    amount = data_dict['AMOUNT'].replace(',', '')
                    data_dict['AMOUNT'] = amount
                df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
    df.head()
    return df
 
def create_database_table():
    # Connect to MySQL database server
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='')
   
    try:
        with connection.cursor() as cursor:
            # Create the database if it doesn't exist
            cursor.execute("DROP DATABASE IF EXISTS invoice_data")
           
            cursor.execute("CREATE DATABASE IF NOT EXISTS invoice_data")
            # Switch to the newly created database
            cursor.execute("USE invoice_data")
            # Create the table if it doesn't exist
            cursor.execute("""CREATE TABLE IF NOT EXISTS invoices (
                                id INT AUTO_INCREMENT PRIMARY KEY,
                                invoice_number VARCHAR(255),
                                description VARCHAR(255),
                                invoice_date VARCHAR(25),
                                due_date VARCHAR(25),
                                amount DECIMAL(10,2),
                                total DECIMAL(10,2),
                                contact_email VARCHAR(255),
                                user_email VARCHAR(255),
                                contact_number VARCHAR(20),
                                bill_to VARCHAR(255)
                            )""")
        connection.commit()
        print("Database and table created successfully!")
    finally:
        connection.close()
       
def insert_into_database(df):
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 database='invoice_data')
    try:
        with connection.cursor() as cursor:
            # Insert data into the table
            for _, row in df.iterrows():
                row = row.where(pd.notnull(row), None)
                sql = """INSERT INTO invoices
                         (invoice_number, description, invoice_date, due_date, amount, total, contact_email, user_email, contact_number, bill_to)
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                cursor.execute(sql, tuple(row))
            connection.commit()
        print("Data inserted into the database successfully!")
    finally:
        connection.close()

def send_email(to_email, subject, message, smtp_server, smtp_port, smtp_username, smtp_password):
    email_message = MIMEMultipart()
    email_message['From'] = smtp_username
    email_message['To'] = to_email
    email_message['Subject'] = subject
    email_message.attach(MIMEText(message, 'plain'))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, to_email, email_message.as_string())
 
def main():
    # load_dotenv()
   
    # st.set_page_config(page_title="Invoice Extraction Bot")
    st.title("Invoice Extraction Bot... ")
    st.subheader("I can help you in extracting invoice data")
 
    # Upload the Invoices (pdf files)
    pdfs = st.file_uploader("Upload invoices here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)
 
     # Create four columns for document upload, data display, chatbot, and email
    tab1,tab2,tab3,tab4 = st.tabs(["Document-Upload", "Get-CSV & AutoStoreDB","Chat-With-Bot","Email"])
   
    global df
   
    with tab1:
        # Upload the Invoices (pdf files)
        # pdfs = st.file_uploader("Upload invoices here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)
 
        if pdfs is None or len(pdfs) == 0:
            st.error("Please select at least one PDF file to upload.")
        else:
            with st.spinner('Extracting data from invoices...'):
                try:
                    df = create_docs(pdfs)
                    st.success("Extracted data successfully!")
                    st.dataframe(df)
                    create_database_table()
                    insert_into_database(df)
                except Exception as e:
                    st.error(f"An error occurred during extraction: {e}")
 
    with tab2:
                df = create_docs(pdfs)
                data_as_csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download data as CSV",
                    data_as_csv,
                    "benchmark-tools.csv",
                    "text/csv",
                    key="download-tools-csv",
                )
 
    with tab3:
                st.header("Chat with Gemini")
                user_question1 = st.text_input("Ask a Question h")
                if user_question1:
                    with st.spinner("Processing..."):
                        response=answer_question(user_question1, data_as_csv.decode('utf-8'))
                        st.write(response)
   
    with tab4:
                # Check if Due Date is available and trigger email notification
                if 'Due Date' in df.columns:
                    due_date_available = df['Due Date'].notnull().any()
                    if due_date_available:
                        if st.button("Send Email Notification"):
                            # Replace 'user_email' with the actual column name containing email addresses
                            user_email =  df['User email'].drop_duplicates()
                            # Replace 'due_date' with the actual column name containing due dates
                            due_date = df['Due Date']
                            st.success("Email notifications sent for due dates.")
                            print(user_email,due_date)
                            for mail in user_email:
                                strr=str(mail)
                                print(type(strr),strr)
                                if strr!="nan":
                                    to_email=str(strr).strip()
                                    subject ="Regarding Bill Due date"
                                    message ="Hi,\n Please pay your bill amount, Due date is approaching \n Thank you"
                                    smtp_server ="smtp.gmail.com"
                                    smtp_port = 587
                                  #replace your email and email app password
                                    smtp_username = "add your email"
                                    smtp_password ="add here password"
                                    send_email(to_email, subject, message, smtp_server, smtp_port, smtp_username, smtp_password)
                            st.success("Email notifications sent for due dates.")
                            
 
    st.success("Hope I was able to save your time❤️")

# Invoking main function
if __name__ == '__main__':
    main()
