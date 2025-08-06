import json
import os
from typing import List, Dict, Optional
from models.embeddings import EmbeddingModel
import logging

logger = logging.getLogger(__name__)

class MentalHealthRAG:
    """
    Mental Health specific RAG system with university counseling knowledge
    """

    def __init__(self, persist_path: str = "mental_health_rag"):
        """Initialize the RAG system"""
        self.embedding_model = EmbeddingModel(persist_path=persist_path)
        self.knowledge_loaded = False
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        """Load the mental health knowledge base"""
        try:
            # Mental health resources and information
            mental_health_docs = [
                {
                    'content': """
                    University Counseling Services at Christ University are available to support students with mental health concerns.
                    Services include individual counseling, group therapy, crisis intervention, and wellness workshops.
                    All counseling services are free and confidential for enrolled students.
                    Hours: Monday to Friday, 9 AM to 5 PM
                    Emergency support is available 24/7 through campus security.
                    """,
                    'metadata': {'type': 'service_info', 'category': 'counseling_services', 'source': 'university_handbook'}
                },

                {
                    'content': """
                    Common signs of depression include persistent sadness, loss of interest in activities,
                    changes in appetite or sleep patterns, fatigue, difficulty concentrating, and feelings of worthlessness.
                    If you experience these symptoms for more than two weeks, it's important to seek professional help.
                    Depression is a treatable medical condition, not a personal weakness.
                    """,
                    'metadata': {'type': 'educational', 'category': 'depression', 'source': 'mental_health_guide'}
                },

                {
                    'content': """
                    Anxiety disorders are characterized by excessive worry, fear, or nervousness that interferes with daily activities.
                    Symptoms may include rapid heartbeat, sweating, trembling, difficulty breathing, and avoidance of situations.
                    Coping strategies include deep breathing exercises, mindfulness meditation, regular exercise, and professional therapy.
                    Cognitive-behavioral therapy (CBT) is particularly effective for anxiety disorders.
                    """,
                    'metadata': {'type': 'educational', 'category': 'anxiety', 'source': 'mental_health_guide'}
                },

                {
                    'content': """
                    Crisis resources for students in immediate danger:
                    - National Suicide Prevention Lifeline: 988 (24/7, free, confidential)
                    - Crisis Text Line: Text HOME to 741741
                    - Emergency services: 911 or campus security
                    - Christ University Campus Security: Available 24/7
                    If you are having thoughts of self-harm, reach out immediately. You are not alone.
                    """,
                    'metadata': {'type': 'crisis_resource', 'category': 'emergency', 'priority': 'high', 'source': 'crisis_guide'}
                },

                {
                    'content': """
                    Stress management techniques for students:
                    1. Time management and organization
                    2. Regular exercise and physical activity
                    3. Adequate sleep (7-9 hours per night)
                    4. Healthy eating habits
                    5. Mindfulness and meditation practices
                    6. Social connections and support systems
                    7. Setting realistic goals and expectations
                    8. Taking regular breaks and practicing self-care
                    """,
                    'metadata': {'type': 'coping_strategies', 'category': 'stress_management', 'source': 'wellness_guide'}
                },

                {
                    'content': """
                    Academic stress is common among university students and can manifest as:
                    - Overwhelming feelings about coursework
                    - Procrastination and avoidance
                    - Perfectionism and fear of failure
                    - Difficulty concentrating or making decisions
                    - Physical symptoms like headaches or fatigue
                    Resources include study skills workshops, time management training, and academic counseling.
                    """,
                    'metadata': {'type': 'educational', 'category': 'academic_stress', 'source': 'student_success_guide'}
                },

                {
                    'content': """
                    Building resilience and emotional wellbeing:
                    - Develop a growth mindset
                    - Practice gratitude and positive thinking
                    - Build strong social connections
                    - Engage in meaningful activities and hobbies
                    - Learn healthy coping mechanisms
                    - Seek help when needed - it's a sign of strength, not weakness
                    """,
                    'metadata': {'type': 'wellness_tips', 'category': 'resilience', 'source': 'wellness_guide'}
                },

                {
                    'content': """
                    Warning signs that indicate you should seek immediate professional help:
                    - Persistent thoughts of death or suicide
                    - Severe depression that interferes with daily functioning
                    - Panic attacks or severe anxiety
                    - Substance abuse or harmful behaviors
                    - Inability to cope with daily stressors
                    - Significant changes in eating or sleeping patterns
                    - Withdrawal from friends, family, and activities
                    """,
                    'metadata': {'type': 'warning_signs', 'category': 'crisis_indicators', 'priority': 'high', 'source': 'crisis_guide'}
                }
            ]

            # Load counselor information as knowledge
            counselors_path = os.path.join(os.path.dirname(__file__), "../resources/counselors.json")
            if os.path.exists(counselors_path):
                with open(counselors_path, 'r') as f:
                    counselors_data = json.load(f)

                for entry in counselors_data:
                    counselor = entry['counselor']
                    programs = entry['programs']

                    content = f"""
                    Counselor: {counselor['name']}
                    Email: {counselor['email']}
                    Phone: {counselor['phone']}
                    Location: {counselor['location']}
                    Programs served: {', '.join(programs)}

                    This counselor provides mental health support and counseling services
                    for students in the specified academic programs.
                    """

                    mental_health_docs.append({
                        'content': content,
                        'metadata': {
                            'type': 'counselor_info',
                            'category': 'staff',
                            'counselor_name': counselor['name'],
                            'programs': programs,
                            'source': 'counselors_directory'
                        }
                    })

            # Add all documents to the embedding model
            self.embedding_model.add_documents(mental_health_docs)
            self.knowledge_loaded = True

            logger.info(f"Loaded {len(mental_health_docs)} mental health knowledge documents")

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self.knowledge_loaded = False

    def retrieve_knowledge(self, query: str, max_results: int = 3, context_length: int = 1500) -> str:
        """
        Retrieve relevant knowledge for a query

        Args:
            query: User query
            max_results: Maximum number of results to include
            context_length: Maximum length of context to return

        Returns:
            Formatted context string
        """
        try:
            if not self.knowledge_loaded:
                self._load_knowledge_base()

            # Get relevant documents
            results = self.embedding_model.search(query, k=max_results, score_threshold=0.2)

            if not results:
                return self._get_default_context(query)

            # Format context with priority for crisis situations
            context_parts = []
            current_length = 0
            crisis_keywords = ['suicide', 'crisis', 'emergency', 'harm', 'danger']

            # Prioritize crisis-related content
            crisis_results = [r for r in results if r['metadata'].get('priority') == 'high' or 
                            any(keyword in query.lower() for keyword in crisis_keywords)]
            other_results = [r for r in results if r not in crisis_results]

            # Process crisis results first
            for result in crisis_results + other_results:
                if current_length >= context_length:
                    break

                text = result['text'].strip()
                metadata = result['metadata']

                # Add source and type information
                source_info = f"[{metadata.get('type', 'info').replace('_', ' ').title()}]"
                if metadata.get('source'):
                    source_info += f" ({metadata['source']})"

                formatted_text = f"{source_info}\n{text}"

                if current_length + len(formatted_text) <= context_length:
                    context_parts.append(formatted_text)
                    current_length += len(formatted_text)
                else:
                    # Try to fit a truncated version
                    remaining_space = context_length - current_length - len(source_info) - 10
                    if remaining_space > 100:
                        truncated_text = text[:remaining_space] + "..."
                        formatted_text = f"{source_info}\n{truncated_text}"
                        context_parts.append(formatted_text)
                    break

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return self._get_default_context(query)

    def _get_default_context(self, query: str) -> str:
        """Get default context when no specific knowledge is found"""
        crisis_keywords = ['suicide', 'crisis', 'emergency', 'harm', 'danger', 'die', 'kill']

        if any(keyword in query.lower() for keyword in crisis_keywords):
            return """
            [Crisis Resource] If you are in immediate danger or having thoughts of self-harm:
            - Call 988 (National Suicide Prevention Lifeline) - available 24/7, free and confidential
            - Text HOME to 741741 (Crisis Text Line)
            - Call 911 or go to your nearest emergency room
            - Contact campus security immediately

            You are not alone, and help is available. These feelings can be temporary, and professional help can make a difference.
            """

        return """
        [General Resource] Christ University provides comprehensive mental health support services.
        All counseling services are free and confidential for students.
        Contact your assigned counselor or visit the counseling office for support.
        Remember: Seeking help is a sign of strength, not weakness.
        """

    def add_custom_knowledge(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add custom knowledge to the RAG system"""
        try:
            if metadata is None:
                metadata = {'type': 'custom', 'source': 'user_added'}

            self.embedding_model.add_text(content, metadata)
            logger.info("Added custom knowledge to RAG system")

        except Exception as e:
            logger.error(f"Error adding custom knowledge: {e}")

    def save_knowledge_base(self) -> None:
        """Save the knowledge base to disk"""
        try:
            self.embedding_model.save_index()
            logger.info("Knowledge base saved successfully")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")

    def get_stats(self) -> Dict:
        """Get RAG system statistics"""
        return self.embedding_model.get_stats()

# Global instance for easy import
mental_health_rag = MentalHealthRAG()

def retrieve_knowledge(query: str, embedding_model=None, max_context_length: int = 1500) -> str:
    """
    Main function for retrieving knowledge (backward compatibility)

    Args:
        query: User query
        embedding_model: Legacy parameter (ignored)
        max_context_length: Maximum context length

    Returns:
        Retrieved context string
    """
    return mental_health_rag.retrieve_knowledge(query, context_length=max_context_length)

def add_knowledge_document(content: str, metadata: Optional[Dict] = None) -> None:
    """Add a new knowledge document to the RAG system"""
    mental_health_rag.add_custom_knowledge(content, metadata)

def save_rag_system() -> None:
    """Save the RAG system to disk"""
    mental_health_rag.save_knowledge_base()
