import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.lookup import find_counselor_by_program
from models.llm import LLMProvider
from utils.rag import mental_health_rag, retrieve_knowledge
from models.embeddings import EmbeddingModel
from utils.rag import retrieve_knowledge
from config.config import OPENAI_API_KEY, GEMINI_API_KEY
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Christ University Mental Health Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_chat_response(chat_model, messages, system_prompt=""):
    """Get response from chat model"""
    try:
        # Convert messages to the format expected by the model
        formatted_messages = []
        if system_prompt:
            formatted_messages.append(SystemMessage(content=system_prompt))
        
        for message in messages:
            if message["role"] == "user":
                formatted_messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                formatted_messages.append(AIMessage(content=message["content"]))
        
        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Error getting response: {str(e)}"
    
def display_counselor_info(counselor):
    """Display counselor details in a clean card-like format"""
    st.markdown(
        f"""
        <div style="background-color:#000000; padding:15px; border-radius:10px; margin:10px 0;">
        <h4> Your Assigned Counselor</h4>
        <p><strong> Name:</strong> {counselor['name']}</p>
        <p><strong> Email:</strong> <a href="mailto:{counselor['email']}">{counselor['email']}</a></p>
        <p><strong> Phone:</strong> <a href="tel:{counselor['phone']}">{counselor['phone']}</a></p>
        <p><strong> Location:</strong> {counselor['location']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# def display_rag_stats():
#     """Display RAG system statistics"""
#     try:
#         # Try to get stats from RAG system
#         from utils.rag import mental_health_rag
#         if hasattr(mental_health_rag, 'get_stats'):
#             stats = mental_health_rag.get_stats()
#         else:
#             stats = {'total_documents': 'Unknown', 'api_available': False, 'ready': False}

#         col1, col2, col3 = st.columns(3)

#         with col1:
#             st.metric("Knowledge Documents", stats.get('total_documents', 0))

#         with col2:
#             api_status = " Ready" if stats.get('ready', False) else "⚠️ Basic Mode"
#             st.metric("Status", api_status)

#         with col3:
#             st.metric("Mode", "Advanced" if stats.get('api_available', False) else "Simple")

#     except Exception as e:
#         st.error(f"Could not load RAG statistics: {e}")

def instructions_page():
    """Instructions and setup page"""
    st.header("📋 Instructions")
    st.markdown("""
    ### Setup Instructions:
    
    1. **API Keys**: Make sure you have your API keys in the `.env` file:
       - `OPENAI_API_KEY` for OpenAI models
       - `GEMINI_API_KEY` for Gemini models
    
    2. **Environment Variables**: The app will automatically load API keys from your `.env` file.
    
    3. **Usage**: 
       - Select your preferred LLM provider (OpenAI or Gemini)
       - Enter your student details
       - Start chatting with the UniCare bot!
    
    ### Features:
    -  Multi-provider LLM support (OpenAI, Gemini)
    -  RAG (Retrieval-Augmented Generation) for context-aware responses
    -  Web search fallback for missing information
    -  Sentiment analysis for better user experience
    -  Counselor lookup by program
    """)



def main():
    st.title("🧠 Christ University Mental Health Support Chatbot")
    st.markdown("*Confidential support for your mental health and wellbeing*")

    # Sidebar for settings and information
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        provider = st.selectbox(
            "Select AI Provider",
            ["OpenAI", "Gemini"],
            help="Choose which AI model to use for responses"
        )

        # Response mode
        response_mode = st.radio(
            "Response Style",
            ["Concise", "Detailed"],
            help="Concise: Brief, focused responses\nDetailed: Comprehensive, in-depth responses"
        )

        # RAG settings
        st.header("📚 Knowledge Base")
        use_rag = st.checkbox("Use Knowledge Base", value=True, help="Retrieve relevant mental health information")

        if use_rag:
            max_context = st.slider("Context Length", 500, 3000, 1500, step=100)

        # Display RAG statistics
        # with st.expander("📊 Knowledge Base Stats"):
        #     display_rag_stats()

        # Crisis resources (always visible)
        st.header("🚨 Crisis Resources")
        st.markdown("""
        <div class="crisis-resources">
        <strong>Immediate Help:</strong><br>
        🆘 National: <strong>988</strong><br>
        💬 Text: <strong>741741</strong><br>
        🚨 Emergency: <strong>911</strong><br>
        🏫 Campus Security: <strong>080-4012-9000</strong>
        </div>
        """, unsafe_allow_html=True)

    # Main content area
    if "student_info" not in st.session_state:
        st.session_state.student_info = {}

    if "chat_enabled" not in st.session_state:
        st.session_state.chat_enabled = False

    # Student Information Section
    if not st.session_state.chat_enabled:
        st.header("📝 Student Information")
        st.markdown("Please provide your details to get personalized support:")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name", placeholder="Enter your full name")
            program = st.text_input("Academic Program", placeholder="e.g., BSc Computer Science")

        with col2:
            reg_number = st.text_input("Registration Number", placeholder="Your student ID")
            year = st.selectbox("Year of Study", ["", "1st Year", "2nd Year", "3rd Year", "4th Year", "Postgraduate"])

        if st.button("Continue to Chat", type="primary"):
            if name and program and reg_number:
                # Store student information
                st.session_state.student_info = {
                    "name": name,
                    "program": program,
                    "reg_number": reg_number,
                    "year": year
                }

                # Find assigned counselor
                try:
                    counselor = find_counselor_by_program(program)
                    st.session_state.counselor = counselor
                except Exception as e:
                    st.session_state.counselor = None
                    logger.warning(f"Could not find counselor: {e}")

                st.session_state.chat_enabled = True
                st.rerun()
            else:
                st.error("Please fill in all required fields (Name, Program, Registration Number)")

    else:
        # Display student info and counselor
        st.header(f"Welcome, {st.session_state.student_info['name']}! 👋")

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.session_state.get('counselor'):
                display_counselor_info(st.session_state.counselor)
            else:
                st.warning(f"No specific counselor found for {st.session_state.student_info['program']}. You can still use the chatbot or contact the general counseling office.")

        with col2:
            if st.button("Start Over", type="secondary"):
                # Reset session
                for key in ['student_info', 'chat_enabled', 'counselor', 'messages']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Chat Interface
        st.header("💬 Mental Health Support Chat")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

            # Welcome message with personalization
            welcome_msg = f"""
            Hello {st.session_state.student_info['name']}! I'm here to provide mental health support and guidance.

            I can help you with:
            \n•  Stress and anxiety management
            \n•  Understanding depression and mood issues  
            \n•  Academic stress and pressure
            \n•  Social and relationship concerns
            \n•  Crisis support and resources
            \n•  Connecting you with appropriate help

            Everything we discuss is confidential. How are you feeling today?
            """

            st.session_state.messages.append({
                "role": "assistant", 
                "content": welcome_msg,
                "timestamp": "now"
            })

        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Initialize LLM
                        llm_provider = LLMProvider(provider=provider)

                        # Get context from RAG if enabled
                        context = ""
                        if use_rag:
                            with st.spinner("Retrieving relevant information..."):
                                context = retrieve_knowledge(prompt, max_context_length=max_context)
                                if context:
                                    st.caption("✅ Found relevant information in knowledge base")

                        # Analyze sentiment (if sentiment.py is implemented)
                        try:
                            sentiment = analyze_sentiment(prompt)
                        except:
                            sentiment = None

                        # Build enhanced system prompt with context
                        system_prompt = f"""
                        You are a mental health support chatbot for Christ University students.

                        Student Information:
                        - Name: {st.session_state.student_info['name']}
                        - Program: {st.session_state.student_info['program']}
                        - Year: {st.session_state.student_info.get('year', 'Not specified')}
                        """

                        if st.session_state.get('counselor'):
                            counselor = st.session_state.counselor
                            system_prompt += f"""

                        Assigned Counselor:
                        - Name: {counselor['name']}
                        - Email: {counselor['email']}
                        - Phone: {counselor['phone']}
                        - Location: {counselor['location']}
                        """

                        if context:
                            system_prompt += f"""

                        Relevant Knowledge Base Information:
                        {context}

                        Use this information to provide accurate, helpful responses. If the knowledge base contains relevant information, incorporate it naturally into your response.
                        """

                        system_prompt += """

                        Guidelines:
                        - Be empathetic, supportive, and non-judgmental
                        - Provide specific, actionable advice when possible
                        - Recognize when professional help is needed
                        - Prioritize crisis situations with immediate resources
                        - Remember this is confidential support
                        - Be culturally sensitive and inclusive
                        """

                        # Generate response
                        response = get_chat_response(
                            llm_provider, 
                            st.session_state.messages[-5:],  # Last 5 messages for context
                            system_prompt
                        )

                        # Display response
                        st.markdown(response)

                        # Add response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "context_used": bool(context),
                            "sentiment": sentiment
                        })

                        # Show context used (for transparency)
                        if use_rag and context:
                            with st.expander("📚 Knowledge Sources Used", expanded=False):
                                st.text_area("Context", context, height=150)

                    except Exception as e:
                        error_msg = f"I apologize, but I'm experiencing technical difficulties. Please try again or contact your counselor directly at {st.session_state.get('counselor', {}).get('email', 'the counseling office')}."
                        st.error(error_msg)
                        logger.error(f"Error generating response: {e}")


# Add custom CSS for better styling
    st.markdown("""
        <style>
            /* Main content text should be white */
            .main .block-container {
                color: #ffffff !important;
            }

            /* Headers and text elements */
            h1, h2, h3, h4, h5, h6 {
                color: #ffffff !important;
            }

            /* Paragraph and div text */
            p, div {
                color: #ffffff !important;
            }

            /* Markdown text */
            .element-container .markdown-text-container {
                color: #ffffff !important;
            }

            /* General text elements */
            .stMarkdown, .stText {
                color: #ffffff !important;
            }
            /* Input field styling - keep dark text in white input fields for readability */
            .stTextInput > div > div > input {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #cccccc !important;
                border-radius: 4px !important;
                padding: 8px !important;
            }

            /* Input field labels - white text */
            .stTextInput > label {
                color: #ffffff !important;
                font-weight: 500 !important;
            }

            /* Fix selectbox styling */
            .stSelectbox > div > div > select {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #cccccc !important;
            }

            /* Selectbox labels - white text */
            .stSelectbox > label {
                color: #ffffff !important;
                font-weight: 500 !important;
            }

            /* Placeholder text */
            .stTextInput input::placeholder {
                color: #666666 !important;
                opacity: 1 !important;
            }

            /* Style the counselor info card */
            .counselor-card {
                background-color: #f0f8ff !important;   
                text-color: #000000 !important;
                border: 1px solid #4CAF50 !important;
                border-radius: 10px !important;
                padding: 15px !important;
                margin: 10px 0 !important;
            }

            /* Counselor card text should be dark for readability */
            .counselor-card p{
                text-color: #000000 !important;
            }
            .counselor-card h4 {
                color: #2c3e50 !important;        
            }
            .counselor-card strong {
                text-color: #000000 !important;        
            }
            .counselor-card a {
                color: #4CAF50 !important;       
            }

            /* Style buttons */
            .stButton > button {
                background-color: #4CAF50 !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 0.5rem 1rem !important;
                font-weight: 500 !important;
                transition: all 0.3s ease !important;
            }

            .stButton > button:hover {
                background-color: #45a049 !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            }

            /* Secondary button styling */
            .stButton > button[kind="secondary"] {
                background-color: #f44336 !important;
                color: white !important;
            }

            .stButton > button[kind="secondary"]:hover {
                background-color: #da190b !important;
            }

            /* Crisis resources styling */
            .crisis-resources {
                background-color: #ffebee !important;
                border-left: 4px solid #f44336 !important;
                padding: 10px !important;
                margin: 10px 0 !important;
                color: #000000 !important;
            }

            /* Chat message styling */
            .stChatMessage {
                background-color: #2c3e50 !important;
                border-radius: 10px !important;
                padding: 10px !important;
                margin: 5px 0 !important;
                color: #ffffff !important;
            }

            /* Metric labels and values */
            .metric-container {
                color: #ffffff !important;
            }

            /* Caption text */
            .caption {
                color: #cccccc !important;
            }

            /* Warning and error text */
            .stAlert {
                color: #000000 !important;
            }

            /* Sidebar keeps its original styling */
            .css-1d391kg {
                background-color: #2c3e50 !important;
            }

            .css-1d391kg .element-container {
                color: #ffffff !important;
            }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
