# Christ University Mental Health Support Chatbot

A comprehensive, AI-powered mental health support system designed specifically for Christ University students. This chatbot provides confidential, empathetic, and context-aware mental health guidance with integrated counselor assignment and crisis intervention capabilities.

## 🌟 Features

### 🤖 **Multi-Provider AI Support**
- **OpenAI GPT-4o-mini**: Advanced reasoning and comprehensive responses
- **Google Gemini 2.0 Flash**: Fast, efficient responses with strong reasoning
- **Provider Selection**: Users can choose their preferred AI model in real-time
- **Response Modes**: Concise (3-4 sentences) or Detailed (5-8 sentences) responses

### 🧠 **Intelligent Mental Health Support**
- **Context-Aware Responses**: Analyzes user messages for crisis indicators, depression, anxiety, and help-seeking patterns
- **Specialized Counseling**: Different response strategies based on detected mental health needs:
  - **Crisis Intervention**: Immediate safety planning and emergency resources
  - **Depression Support**: Behavioral activation, sleep hygiene, and recovery strategies
  - **Anxiety Management**: Grounding techniques, breathing exercises, and CBT approaches
  - **Help-Seeking Guidance**: Clear pathways to professional support

### 📚 **RAG-Powered Knowledge Base**
- **Retrieval-Augmented Generation**: Combines AI responses with curated mental health knowledge
- **University-Specific Content**: Christ University counseling services, policies, and resources
- **Crisis Resources**: Immediate access to emergency hotlines and campus security
- **Educational Content**: Depression, anxiety, stress management, and academic support information

### 👥 **Counselor Assignment System**
- **Program-Based Matching**: Automatically assigns counselors based on academic programs
- **Contact Information**: Direct access to counselor emails, phone numbers, and office locations
- **13 Counselors**: Comprehensive coverage across all university programs
- **Real-time Lookup**: Instant counselor information display

### 🎨 **Modern User Interface**
- **Streamlit Web App**: Clean, responsive interface with dark theme
- **Personalized Experience**: Student information collection and personalized welcome
- **Chat Interface**: Real-time conversation with message history
- **Crisis Resources**: Always-visible emergency contact information
- **Knowledge Transparency**: Optional display of information sources used

### 🔒 **Privacy & Security**
- **Confidential Support**: All conversations are private and not stored permanently
- **Crisis Recognition**: Automatic detection of emergency situations
- **Immediate Resources**: Direct access to crisis hotlines and emergency services
- **Professional Boundaries**: Clear guidance on when to seek professional help

## 🏗️ Architecture

### **Core Components**

```
mental-health-chatbot/
├── app.py                          # Main Streamlit application
├── config/
│   └── config.py                   # API key management
├── models/
│   ├── llm.py                      # Multi-provider LLM interface
│   └── embeddings.py               # FAISS-based embedding system
├── utils/
│   ├── rag.py                      # RAG system implementation
│   └── lookup.py                   # Counselor assignment logic
├── resources/
│   └── counselors.json             # Counselor database
├── initialize_rag.py               # Knowledge base initialization
└── requirements.txt                # Python dependencies
```

### **Technology Stack**
- **Frontend**: Streamlit (Python web framework)
- **AI Models**: OpenAI GPT-4o-mini, Google Gemini 2.0 Flash
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Language Processing**: LangChain Core
- **Data Storage**: JSON files with pickle persistence

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8 or higher
- OpenAI API key
- Google Gemini API key (optional)

### **Installation**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mental-health-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Initialize the knowledge base**
   ```bash
   python initialize_rag.py
   ```

5. **Launch the application**
   ```bash
   streamlit run app.py
   ```

### **First Run Setup**

1. **Access the application** at `http://localhost:8501`
2. **Enter student information**:
   - Full Name
   - Academic Program
   - Registration Number
   - Year of Study
3. **Select AI provider** (OpenAI or Gemini)
4. **Choose response style** (Concise or Detailed)
5. **Start chatting** with the mental health support bot

## 📋 Detailed Setup Instructions

### **API Key Configuration**

The application supports two AI providers:

#### **OpenAI Setup**
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and generate an API key
3. Add to `.env` file: `OPENAI_API_KEY=sk-...`

#### **Google Gemini Setup**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to `.env` file: `GEMINI_API_KEY=...`

### **Knowledge Base Initialization**

The RAG system requires initialization to create the vector database:

```bash
python initialize_rag.py
```

This process:
- Loads mental health knowledge documents
- Creates embeddings using OpenAI
- Builds FAISS index for similarity search
- Saves the knowledge base to disk

### **Counselor Database**

The system includes a comprehensive counselor database covering all Christ University programs:

- **13 Counselors** across different academic blocks
- **Program-specific assignments** for personalized support
- **Contact information** including email, phone, and office location
- **Real-time lookup** based on student's academic program

## 🎯 Usage Guide

### **Student Onboarding**

1. **Information Collection**
   - Provide full name, academic program, registration number
   - System automatically assigns appropriate counselor
   - Counselor contact information displayed prominently

2. **AI Provider Selection**
   - Choose between OpenAI and Gemini
   - Switch providers during conversation if needed
   - Compare response quality and style

3. **Response Customization**
   - **Concise Mode**: Brief, focused responses (3-4 sentences)
   - **Detailed Mode**: Comprehensive support (5-8 sentences)
   - Adjust based on preference and urgency

### **Mental Health Support Features**

#### **Crisis Intervention**
- **Automatic Detection**: Recognizes crisis keywords and suicidal thoughts
- **Immediate Response**: Provides emergency resources and safety planning
- **Crisis Resources**: Always visible emergency contact information
- **Safety Planning**: Step-by-step guidance for immediate safety

#### **Depression Support**
- **Symptom Recognition**: Identifies depressive symptoms and patterns
- **Behavioral Activation**: Suggests small, achievable tasks
- **Sleep Hygiene**: Guidance on establishing healthy sleep patterns
- **Professional Referral**: Clear pathways to counseling services

#### **Anxiety Management**
- **Grounding Techniques**: 5-4-3-2-1 method and breathing exercises
- **Trigger Identification**: Helps identify anxiety sources
- **Coping Strategies**: Progressive muscle relaxation and mindfulness
- **Lifestyle Guidance**: Exercise, sleep, and caffeine management

#### **Academic Stress Support**
- **Study Skills**: Time management and organization strategies
- **Perfectionism**: Addressing unrealistic expectations
- **Procrastination**: Breaking down overwhelming tasks
- **Work-Life Balance**: Maintaining healthy boundaries

### **Knowledge Base Integration**

The RAG system provides context-aware responses by:

1. **Query Analysis**: Understanding user's mental health needs
2. **Knowledge Retrieval**: Finding relevant information from curated database
3. **Context Integration**: Combining AI responses with factual knowledge
4. **Source Transparency**: Optional display of information sources

### **Counselor Assignment**

The system automatically:

1. **Matches Programs**: Links student's program to assigned counselor
2. **Displays Contact Info**: Shows counselor's email, phone, and office location
3. **Provides Direct Access**: Clickable email and phone links
4. **Fallback Support**: General counseling office information if no specific match

## 🔧 Configuration Options

### **RAG Settings**
- **Knowledge Base Toggle**: Enable/disable RAG integration
- **Context Length**: Adjust from 500-3000 characters
- **Search Results**: Configure number of relevant documents (1-5)
- **Similarity Threshold**: Fine-tune relevance scoring (0.1-0.9)

### **AI Model Parameters**
- **Temperature**: Control response creativity (0.0-1.0)
- **Max Tokens**: Limit response length
- **Model Selection**: Choose specific model versions
- **Provider Fallback**: Automatic switching on API errors

### **Interface Customization**
- **Theme**: Dark/light mode options
- **Layout**: Wide/centered layout preferences
- **Sidebar**: Collapsible settings panel
- **Chat History**: Configurable message retention

## 🛠️ Development

### **Project Structure**

```
mental-health-chatbot/
├── app.py                          # Main application entry point
├── config/
│   └── config.py                   # Environment and API configuration
├── models/
│   ├── llm.py                      # LLM provider abstraction
│   └── embeddings.py               # Vector embedding system
├── utils/
│   ├── rag.py                      # RAG implementation
│   └── lookup.py                   # Counselor lookup utilities
├── resources/
│   └── counselors.json             # Counselor database
├── mental_health_rag/              # Persistent RAG data
├── initialize_rag.py               # Knowledge base setup
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

### **Adding New Features**

#### **Extending Knowledge Base**
```python
from utils.rag import add_knowledge_document

# Add new mental health content
add_knowledge_document(
    content="New mental health information...",
    metadata={'type': 'educational', 'category': 'new_topic'}
)
```

#### **Adding New Counselors**
Edit `resources/counselors.json`:
```json
{
  "programs": ["New Program"],
  "counselor": {
    "name": "Counselor Name",
    "email": "email@christuniversity.in",
    "phone": "080-4012-XXXX",
    "location": "Block, Floor, Cabin"
  }
}
```

#### **Customizing AI Responses**
Modify `models/llm.py` to add new response patterns:
```python
def _build_prompt(self, query, context, history, mode):
    # Add new keyword detection
    new_keywords = ['your_keywords']
    # Add new system prompt for detected patterns
```

### **Testing**

#### **Unit Tests**
```bash
# Test RAG system
python -c "from utils.rag import mental_health_rag; print(mental_health_rag.get_stats())"

# Test counselor lookup
python -c "from utils.lookup import find_counselor_by_program; print(find_counselor_by_program('BSc Computer Science'))"

# Test LLM provider
python -c "from models.llm import LLMProvider; llm = LLMProvider('OpenAI'); print('LLM initialized successfully')"
```

#### **Integration Testing**
```bash
# Run the full application
streamlit run app.py

# Test knowledge base initialization
python initialize_rag.py
```

## 🚨 Crisis Resources

### **Immediate Emergency Contacts**
- **National Suicide Prevention Lifeline**: 988 (24/7, free, confidential)
- **Crisis Text Line**: Text HOME to 741741
- **Emergency Services**: 911
- **Campus Security**: 080-4012-9000

### **Christ University Support**
- **Counseling Office**: Available Monday-Friday, 9 AM-5 PM
- **Emergency Support**: 24/7 through campus security
- **Confidential Services**: Free for all enrolled students

## 🤝 Contributing

### **Guidelines**
1. **Mental Health Sensitivity**: All contributions must prioritize student safety and wellbeing
2. **Crisis Awareness**: Maintain clear pathways to professional help
3. **Cultural Sensitivity**: Respect diverse backgrounds and experiences
4. **Privacy Protection**: Ensure all data handling follows privacy best practices

### **Development Workflow**
1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate testing
4. Submit a pull request with detailed description
5. Ensure all crisis intervention features remain functional

## 📄 License

This project is developed for Christ University's mental health support services. All rights reserved.

## 🙏 Acknowledgments

- **Christ University Counseling Services** for program-specific guidance
- **OpenAI** and **Google** for AI model access
- **Streamlit** for the web application framework
- **FAISS** for efficient vector similarity search
- **LangChain** for AI/LLM integration tools

## 📞 Support

For technical support or questions about the mental health chatbot:

- **Technical Issues**: Check the setup instructions and configuration
- **Mental Health Support**: Contact your assigned counselor or the counseling office
- **Crisis Situations**: Use the emergency resources listed above

---

**⚠️ Important**: This chatbot is designed to provide mental health support and guidance but is not a replacement for professional mental health care. In crisis situations, always contact emergency services or professional mental health providers immediately.