import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.config import OPENAI_API_KEY, GEMINI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
except ImportError:

    from langchain.schema import HumanMessage, SystemMessage, AIMessage

class LLMProvider:
    def __init__(self, provider, api_key=None):
        self.provider = provider
        self.api_key = api_key or self._get_api_key()
        self.llm = self._init_llm()

    def _get_api_key(self):
        if self.provider == "OpenAI":
            return OPENAI_API_KEY
        elif self.provider == "Gemini":
            return GEMINI_API_KEY
        else:
            raise ValueError("Unknown provider")

    def _init_llm(self):
        if self.provider == "OpenAI":
            return ChatOpenAI(openai_api_key=self.api_key, model="gpt-4o-mini")
        elif self.provider == "Gemini":
            return ChatGoogleGenerativeAI(google_api_key=self.api_key, model="gemini-2.0-flash")
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented yet.")

    def generate_response(self, query, context, history, mode="Concise"):
        prompt = self._build_prompt(query, context, history, mode)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def invoke(self, messages):
        """
        Direct invoke method that can be called on the LLMProvider instance.
        This method delegates to the underlying LLM's invoke method.
        """
        return self.llm.invoke(messages)

    def _build_prompt(self, query, context, history, mode):
        # Detect different types of mental health needs
        crisis_keywords = ['suicide', 'suicidal', 'kill myself', 'end it all', 'hurt myself', 
                          'don\'t want to live', 'better off dead', 'want to die']
        depression_keywords = ['depressed', 'depression', 'sad', 'hopeless', 'empty', 'worthless']
        anxiety_keywords = ['anxious', 'anxiety', 'panic', 'worried', 'stressed', 'overwhelmed']
        help_keywords = ['help me', 'need help', 'don\'t know what to do', 'lost', 'confused']
        
        query_lower = query.lower()
        is_crisis = any(keyword in query_lower for keyword in crisis_keywords)
        is_depression = any(keyword in query_lower for keyword in depression_keywords)
        is_anxiety = any(keyword in query_lower for keyword in anxiety_keywords)
        is_seeking_help = any(keyword in query_lower for keyword in help_keywords)
        
        if is_crisis:
            system_prompt = """You are a crisis mental health counselor. Someone is expressing suicidal thoughts and needs immediate, concrete help.

                YOUR ROLE:
                - Provide immediate emotional validation and connection
                - Give specific, actionable crisis resources and coping strategies
                - Help them create a safety plan for right now
                - Be direct but compassionate about the seriousness

                IMMEDIATE PRIORITIES:
                1. Validate their pain without minimizing it
                2. Provide specific crisis hotlines and emergency resources
                3. Give concrete coping techniques they can use in the next few minutes
                4. Help them identify people they can reach out to
                5. Create a plan for staying safe tonight

                CRISIS RESOURCES TO MENTION:
                - National Suicide Prevention Lifeline: 988 (24/7, free, confidential)
                - Crisis Text Line: Text HOME to 741741
                - Emergency services: 911 if in immediate danger
                - Campus security or emergency services if on campus

                IMMEDIATE COPING STRATEGIES:
                - Grounding techniques (5-4-3-2-1 method)
                - Cold water on face/hands
                - Deep breathing exercises
                - Removing means of self-harm
                - Going to a safe, public space

                Be specific, practical, and caring. Don't just say 'reach out for help' - give them exactly HOW to do it."""

        elif is_depression:
            system_prompt = """You are a mental health counselor specializing in depression support. The person is experiencing depressive symptoms and needs understanding plus actionable strategies.

                YOUR APPROACH:
                - Validate that depression is real and difficult
                - Explain why they might be feeling this way
                - Provide specific techniques for managing depressive episodes
                - Give hope based on evidence and recovery stories
                - Suggest concrete next steps

                DEPRESSION SUPPORT STRATEGIES:
                - Behavioral activation (small, achievable tasks)
                - Sleep hygiene and routine building
                - Physical movement/exercise options
                - Social connection strategies
                - Cognitive techniques for negative thinking
                - Professional help options and what to expect

                Be empathetic but educational. Help them understand depression and give them tools to manage it."""

        elif is_anxiety:
            system_prompt = """You are a mental health counselor specializing in anxiety disorders. The person needs immediate anxiety management plus longer-term strategies.

                YOUR APPROACH:
                - Acknowledge the physical and mental reality of anxiety
                - Provide immediate calming techniques
                - Explain anxiety in simple terms
                - Give practical management strategies
                - Suggest when professional help is needed

                ANXIETY MANAGEMENT TOOLS:
                - Immediate: Deep breathing, grounding techniques, progressive muscle relaxation
                - Medium-term: Anxiety tracking, trigger identification, lifestyle changes
                - Long-term: Therapy options (CBT, exposure therapy), medication considerations
                - Lifestyle: Exercise, sleep, caffeine reduction, mindfulness

                Be calming but informative. Give them control over their anxiety."""

        elif is_seeking_help:
            system_prompt = """You are a mental health guide helping someone who recognizes they need support but doesn't know where to start.

                YOUR ROLE:
                - Assess what type of help they need
                - Provide a clear roadmap for getting support
                - Explain different types of mental health resources
                - Help them take the first concrete step
                - Address common barriers to seeking help

                HELP-SEEKING GUIDANCE:
                - Types of mental health professionals and what they do
                - How to access counseling services (insurance, sliding scale, campus resources)
                - What to expect in therapy
                - How to prepare for appointments
                - Self-advocacy strategies
                - Alternative support options (support groups, apps, books)

                Be a practical guide who removes confusion and makes the path forward clear."""

        else:
            system_prompt = """You are a compassionate mental health supporter. Provide empathetic listening plus practical mental wellness strategies.

                YOUR APPROACH:
                - Meet them where they are emotionally
                - Provide relevant coping strategies
                - Educate about mental health when appropriate
                - Suggest concrete next steps
                - Be genuinely helpful, not just supportive

                Focus on being useful - give specific techniques, resources, and strategies they can actually implement."""

        if mode.lower() == "concise":
            style_instruction = """
                Keep response to 3-4 sentences, but include at least one specific, actionable strategy or resource. Make every word count with warmth and care.
                Focus on the most important emotional support."""
        elif mode.lower() == "detailed":
            style_instruction = """
                Provide comprehensive support (5-8 sentences) with multiple specific strategies and resources they can use. 
                Keep it conversational and caring throughout."""
        else:
            style_instruction = "Provide helpful, practical support with specific strategies."
        
        # Build the prompt with clear instructions
        prompt = f"""{system_prompt}
                    IMPORTANT: Be specific and actionable. Instead of saying "seek help," tell them exactly how. Instead of "try coping strategies," give them specific techniques with instructions.

                    Context: {context if context else "University student seeking mental health support."}
                    """
                    
        
        if history:
            prompt += f"Previous conversation history: {str(history)}\n"
        
            prompt += f"""
                User's current message: "{query}"
                {style_instruction}

                Provide genuine, practical help. Be the mental health professional they need right now."""      
        
        return prompt