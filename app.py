import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import base64
from io import BytesIO

st.set_page_config(
    page_title="Ask Your CSV",
    page_icon="📊",
    layout="wide"
)

# Configure Gemini API key from Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.0-flash')

# Helper function to export conversation as HTML report
def export_conversation():
    if not st.session_state.messages:
        return None
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>-
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 30px; }}
            .question {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .answer {{ padding: 10px; margin: 10px 0; }}
            .metadata {{ color: #999; font-size: 14px; }}
            code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>Data Analysis Report</h1>
        <p class="metadata">Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    """
    
    if st.session_state.df is not None:
        html_content += f"""
        <h2>Dataset Information</h2>
        <ul>
            <li>Total Rows: {st.session_state.df.shape[0]}</li>
            <li>Total Columns: {st.session_state.df.shape[1]}</li>
            <li>Columns: {', '.join(st.session_state.df.columns)}</li>
        </ul>
        """
    
    html_content += "<h2>Analysis Conversation</h2>"
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            html_content += f'<div class="question"><strong>Question:</strong> {msg["content"]}</div>'
        else:
            content = msg["content"].replace("```python", "<pre><code>").replace("```", "</code></pre>")
            html_content += f'<div class="answer"><strong>Analysis:</strong><br>{content}</div>'
            if "figure" in msg:
                html_content += '<p><em>[Visualization generated - see application for details]</em></p>'
    
    html_content += "</body></html>"
    return html_content

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "data_summary" not in st.session_state:
    st.session_state.data_summary = None

st.title("📊 Ask Your CSV")
st.markdown("Upload your data and ask questions in plain English!")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Store summary info for prompt (token optimization)
            st.session_state.data_summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "sample": df.head(3).to_dict(),
                "stats": df.describe().to_dict() if not df.empty else {}
            }
            
            st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            
            with st.expander("Preview Data"):
                st.dataframe(df.head())
            
            with st.expander("Data Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", df.shape[0])
                    st.metric("Total Columns", df.shape[1])
                with col2:
                    st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
                    st.metric("Missing Values", df.isnull().sum().sum())
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV format.")
    else:
        st.info("👆 Upload a CSV file to start analyzing!")
    
    # Export options (only show if there are messages)
    if st.session_state.messages:
        st.markdown("---")
        st.header("💾 Export Options")
        if st.button("Generate Report"):
            export_html = export_conversation()
            st.download_button(
                label="📥 Download Report (HTML)",
                data=export_html,
                file_name=f"data_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html"
            )
            st.info("💡 Tip: Open the HTML file and print to PDF for best results")

# Main chat interface
if st.session_state.df is not None:
    # Show chat history with figures
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "figure" in msg:
                st.pyplot(msg["figure"])
    
    user_input = st.chat_input("Ask a question about your data")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        #prepare data contact with token optimization
        df = st.session_state.df
        
        # Build data context for prompt (use summary if large)
        if len(df) > 100:
            data_context = f"""
Dataset shape: {st.session_state.data_summary['shape']}
Columns: {', '.join(st.session_state.data_summary['columns'])}
Data types: {st.session_state.data_summary['dtypes']}
Sample rows: {st.session_state.data_summary['sample']}
Basic statistics: {st.session_state.data_summary['stats']}
"""
        else:
            data_context = f"Full dataset:\n{df.to_string()}"
        
        # Build prompt from last few messages + system instructions + data context
        system_prompt = (
            "You are a helpful data analyst assistant.\n\n"
            "The user has uploaded a CSV file with the following information:\n"
            f"{data_context}\n\n"
            "The data is loaded in a pandas DataFrame called 'df'.\n"
            "Guidelines:\n"
            "- Answer the user's question clearly and concisely.\n"
            "- If analysis is needed, write Python code using pandas, matplotlib, seaborn.\n"
            "- Use plt.figure() before plotting and plt.tight_layout() to format charts.\n"
            "- Validate data before operations.\n"
            "- If unsure or limited by data, explain.\n"
            "- Focus on results and insights, not code details.\n"
            "- Show charts inline if relevant.\n"
           
        )
        
        # Collect last 6 messages for context (user + assistant)
        recent_msgs = st.session_state.messages[-6:]
        conversation_text = ""
        for msg in recent_msgs:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            # Truncate long content to save tokens
            if len(content) > 1000:
                content = content[:1000] + "..."
            conversation_text += f"{role}: {content}\n\n"
        
        prompt = system_prompt + "\nConversation:\n" + conversation_text + f"User: {user_input}\nAssistant:"
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Analyzing your data..."):
                try:
                    response = model.generate_content(prompt)
                    reply = response.text.strip()
                    placeholder.markdown(reply)
                    
                    # Execute Python code if present
                    if "```python" in reply:
                        code_blocks = reply.split("```python")
                        for i in range(1, len(code_blocks)):
                            code = code_blocks[i].split("```")[0]
                            try:
                                with warnings.catch_warnings(record=True) as w:
                                    warnings.simplefilter("always")
                                    
                                    plt.figure(figsize=(10, 6))
                                    exec_globals = {
                                        "df": df,
                                        "pd": pd,
                                        "plt": plt,
                                        "sns": sns,
                                        "st": st
                                    }
                                    exec(code.strip(), exec_globals)
                                    
                                    if w:
                                        for warning in w:
                                            st.info(f"Note: {warning.message}")
                                    
                                    fig = plt.gcf()
                                    if fig.get_axes():
                                        st.pyplot(fig)
                                        # Save figure in message history for persistence
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": reply,
                                            "figure": fig
                                        })
                                    else:
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": reply
                                        })
                                    
                                    plt.close()
                            except Exception as e:
                                error_type = type(e).__name__
                                st.error(f"Code execution failed: {error_type}")
                                if "NameError" in str(e):
                                    st.info("This might mean a column name is misspelled or doesn't exist.")
                                elif "TypeError" in str(e):
                                    st.info("This often happens when trying to plot non-numeric data.")
                                elif "KeyError" in str(e):
                                    st.info("The specified column might not exist in your dataset.")
                                else:
                                    st.info("Try rephrasing your question or check your data format.")
                                st.code(code, language="python")
                    else:
                        # No code detected, just append text
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": reply
                        })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Please upload a CSV file to start")
        st.markdown("### 💡 Example questions you can ask:")
        st.markdown("""
        - What are the main trends in my data?
        - Show me a correlation matrix
        - Create a bar chart of the top 10 categories
        - What's the average value by month?
        - Are there any outliers in the price column?
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
💡 Tip: Be specific with your questions for better results | 🔒 Your data stays private and is not stored
</div>
""", unsafe_allow_html=True)
