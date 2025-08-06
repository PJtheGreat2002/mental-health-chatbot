import os
import sys
import json
from pathlib import Path
import config.config

try:
    from models.embeddings import EmbeddingModel
    from utils.rag import MentalHealthRAG, mental_health_rag
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    def initialize_knowledge_base():
        """Initialize the RAG knowledge base"""
        try:
            print("Initializing Christ University Mental Health ChatBot RAG System...")
            print("=" * 60)

            print("Loading mental health knowledge base...")

            # The knowledge base is automatically loaded when MentalHealthRAG is initialized
            # Let's verify it worked
            stats = mental_health_rag.get_stats()

            print(f" Knowledge base initialized successfully!")
            print(f" Statistics:")
            print(f"   - Total documents: {stats['total_documents']}")
            print(f"   - Vector dimension: {stats['dimension']}")
            print(f"   - Index size: {stats['index_size']}")

            if stats.get('metadata_keys'):
                print(f"   - Metadata categories: {', '.join(stats['metadata_keys'])}")

            # Test the retrieval system
            print("\nTesting retrieval system...")
            test_queries = [
                "I'm feeling very anxious about exams",
                "How can I contact a counselor?",
                "I'm having thoughts of suicide",
                "What are signs of depression?"
            ]

            for query in test_queries:
                print(f"\n   Query: '{query}'")
                context = mental_health_rag.retrieve_knowledge(query, max_results=2, context_length=300)
                if context:
                    print(f"    Retrieved context ({len(context)} characters)")
                else:
                    print(f"    No context retrieved")

            # Save the knowledge base
            print("\n Saving knowledge base to disk...")
            mental_health_rag.save_knowledge_base()

            print("\nRAG system initialization complete!")
            print("\nYou can now run the chatbot with: streamlit run app.py")

        except Exception as e:
            print(f" Error initializing knowledge base: {e}")
            print("\nPlease ensure you have:")
            print("1. Set up your OpenAI API key in the .env file or config.py")
            print("2. Installed all required dependencies: pip install -r requirements.txt")
            print("3. Have the counselors.json file in the resources/ directory")
            return False

        return True

    def verify_setup():
        """Verify that the setup is correct"""
        print(" Verifying setup...")

        # Check for required files
        required_files = [
            'config/config.py',
            'models/embeddings.py',
            'utils/rag.py',
            'resources/counselors.json'
        ]

        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            print(f" Missing required files: {', '.join(missing_files)}")
            return False

        # Check for API keys
        try:
            from config.config import OPENAI_API_KEY, GEMINI_API_KEY
            openai_key = os.getenv('OPENAI_API_KEY')
            gemini_key = os.getenv('GEMINI_API_KEY')
            if not openai_key:
                print(" OpenAI API key not found in config")
                return False
            if not gemini_key:
                print(" Gemini API key not found in config")
        except ImportError:
            print(" Could not import config.py")
            return False

        print(" Setup verification passed")
        return True

    if __name__ == "__main__":
        print(" Christ University Mental Health ChatBot - RAG Initialization")
        print("=" * 60)

        if verify_setup():
            if initialize_knowledge_base():
                print("\n✨ Ready to help students with mental health support!")
            else:
                sys.exit(1)
        else:
            print("\nPlease fix the setup issues and try again.")
            sys.exit(1)

except ImportError as e:
    print(f" Import error: {e}")
    print("\nPlease install the required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)
