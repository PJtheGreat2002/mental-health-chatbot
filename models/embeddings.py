from langchain_openai import OpenAIEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import json
import os
import pickle
from typing import List, Dict, Optional, Tuple
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Enhanced embedding model with document processing, persistence, and better search capabilities
    """

    def __init__(self, model_name: str = "text-embedding-ada-002", persist_path: str = "embeddings_index"):
        """
        Initialize the embedding model

        Args:
            model_name: OpenAI embedding model name
            persist_path: Path to persist the FAISS index and metadata
        """
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
            self.dimension = 1536  # OpenAI text-embedding-ada-002 dimension
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []
            self.metadata = []
            self.persist_path = persist_path
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            # Try to load existing index
            self._load_index()

            logger.info(f"EmbeddingModel initialized with {len(self.texts)} existing documents")

        except Exception as e:
            logger.error(f"Error initializing EmbeddingModel: {e}")
            raise

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            if not text or not text.strip():
                return np.zeros(self.dimension, dtype='float32')

            embedding = self.embeddings.embed_query(text.strip())
            return np.array(embedding).astype('float32')
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return np.zeros(self.dimension, dtype='float32')

    def add_text(self, text: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a single text with optional metadata

        Args:
            text: Text to add
            metadata: Optional metadata dictionary
        """
        try:
            if not text or not text.strip():
                return

            vec = self.embed(text.strip())
            if np.any(vec):  # Only add non-zero embeddings
                self.index.add(np.array([vec]))
                self.texts.append(text.strip())
                self.metadata.append(metadata or {})
                logger.debug(f"Added text: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error adding text: {e}")

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add multiple documents with chunking

        Args:
            documents: List of dictionaries with 'content' and optional 'metadata'
        """
        try:
            for doc in documents:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})

                if content:
                    # Split document into chunks
                    chunks = self.text_splitter.split_text(content)

                    for i, chunk in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata['chunk_id'] = i
                        chunk_metadata['total_chunks'] = len(chunks)
                        self.add_text(chunk, chunk_metadata)

            logger.info(f"Added {len(documents)} documents with chunking")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")

    def search(self, query: str, k: int = 5, score_threshold: float = 0.8) -> List[Dict]:
        """
        Search for similar texts with metadata and scores

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1, higher is more similar)

        Returns:
            List of dictionaries with text, metadata, and similarity scores
        """
        try:
            if not query.strip() or len(self.texts) == 0:
                return []

            vec = self.embed(query.strip())
            if not np.any(vec):
                return []

            # Search in FAISS index
            distances, indices = self.index.search(np.array([vec]), min(k, len(self.texts)))

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.texts) and idx >= 0:
                    # Convert L2 distance to similarity score (0-1)
                    similarity = max(0, 1 - (distance / 4.0))  # Normalize distance

                    if similarity >= score_threshold:
                        results.append({
                            'text': self.texts[idx],
                            'metadata': self.metadata[idx],
                            'similarity': float(similarity),
                            'distance': float(distance)
                        })

            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)

            logger.debug(f"Found {len(results)} results for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def get_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """
        Get relevant context as a formatted string for RAG

        Args:
            query: Search query
            max_context_length: Maximum length of returned context

        Returns:
            Formatted context string
        """
        try:
            results = self.search(query, k=5, score_threshold=0.3)

            if not results:
                return ""

            context_parts = []
            current_length = 0

            for result in results:
                text = result['text']
                metadata = result.get('metadata', {})

                # Add source information if available
                source_info = ""
                if metadata.get('source'):
                    source_info = f"[Source: {metadata['source']}] "
                elif metadata.get('type'):
                    source_info = f"[{metadata['type']}] "

                formatted_text = f"{source_info}{text}"

                if current_length + len(formatted_text) <= max_context_length:
                    context_parts.append(formatted_text)
                    current_length += len(formatted_text)
                else:
                    break

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return ""

    def save_index(self) -> None:
        """Save the FAISS index and metadata to disk"""
        try:
            os.makedirs(self.persist_path, exist_ok=True)

            # Save FAISS index
            index_path = os.path.join(self.persist_path, "index.faiss")
            faiss.write_index(self.index, index_path)

            # Save texts and metadata
            data = {
                'texts': self.texts,
                'metadata': self.metadata,
                'dimension': self.dimension
            }

            data_path = os.path.join(self.persist_path, "data.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Index saved to {self.persist_path}")

        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def _load_index(self) -> None:
        """Load the FAISS index and metadata from disk"""
        try:
            index_path = os.path.join(self.persist_path, "index.faiss")
            data_path = os.path.join(self.persist_path, "data.pkl")

            if os.path.exists(index_path) and os.path.exists(data_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)

                # Load texts and metadata
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)

                self.texts = data.get('texts', [])
                self.metadata = data.get('metadata', [])
                self.dimension = data.get('dimension', self.dimension)

                logger.info(f"Loaded existing index with {len(self.texts)} documents")
            else:
                logger.info("No existing index found, starting fresh")

        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            # Reset to empty state
            self.texts = []
            self.metadata = []
            self.index = faiss.IndexFlatL2(self.dimension)

    def get_stats(self) -> Dict:
        """Get statistics about the embedding model"""
        return {
            'total_documents': len(self.texts),
            'index_size': self.index.ntotal,
            'dimension': self.dimension,
            'metadata_keys': list(set().union(*(d.keys() for d in self.metadata if d)))
        }

    def clear_index(self) -> None:
        """Clear all data from the index"""
        self.texts = []
        self.metadata = []
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info("Index cleared")
