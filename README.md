# -Invoice-Data-Extraction-and-Notification-System
The "Invoice Data Extraction and Notification System" is a Streamlit web application designed to automate the extraction of key information from PDF invoices and notify users about pending payment due dates via email.

Features:

Document Upload: Users can upload PDF invoice files containing billing information.
Data Extraction: The system automatically extracts relevant details such as invoice number, description, invoice date, due date, amount, total, contact email, user email, contact number, and billing address from the uploaded invoices.
CSV Export: Users can download the extracted data as a CSV file for further analysis or record-keeping.
Chatbot Interaction: Users can interact with a conversational chatbot named "Gemini" to ask questions related to the extracted invoice data.
Email Notifications: The system identifies pending payment due dates from the extracted data and sends email notifications to the respective users, reminding them to pay their bills.
Workflow:

Users upload one or more PDF invoices containing billing information.
The system extracts relevant data from the invoices, processes it, and displays it in a tabular format.
Users can download the extracted data as a CSV file for their records.
Users can interact with the chatbot to inquire about specific details or ask questions related to the extracted data.
The system identifies pending payment due dates from the extracted data and sends email notifications to the users reminding them to pay their bills.
Technologies Used:

Streamlit: for building the interactive web application interface.
PyPDF2: for parsing PDF documents and extracting text.
Langchain: for natural language processing tasks such as text splitting and conversational AI.
Pandas: for data manipulation and analysis.
PyMySQL: for interacting with MySQL database to store the extracted data.
smtplib: for sending email notifications.
dotenv: for loading environment variables securely.
Benefits:

Automation: The system automates the tedious task of manually extracting invoice data, saving time and effort.
Accuracy: By leveraging natural language processing techniques, the system ensures accurate extraction of key information from invoices.
Convenience: Users can conveniently access and download the extracted data, as well as interact with the chatbot for additional insights.
Timely Notifications: The email notification feature helps users stay informed about pending payment due dates, reducing the risk of late payments and associated penalties.
Potential Enhancements:

Support for more invoice formats and languages.
Integration with cloud storage platforms for seamless document management.
Customizable email templates and scheduling options for notifications.
Advanced analytics and visualization capabilities for deeper insights into billing patterns and trends.
