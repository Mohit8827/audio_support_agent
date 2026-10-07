"""
LLM Agent with RAG Tool Integration

This module defines the customer support agent that uses a Language Model
with Retrieval-Augmented Generation (RAG) capabilities.

Students should implement the RAG tool function and complete the agent setup.
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import Optional, Dict, Any, List


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for LLM agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.
        
        Args:
            config: Configuration dictionary containing LLM settings, prompts, etc.
        """
        self.config = config or {}
        self.is_initialized = False
        self.chat_history = []
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent with LLM and tools."""
        pass
    
    @abstractmethod
    async def process_query(self, text: str, **kwargs) -> str:
        """
        Process a text query and return a response.
        
        Args:
            text: Input text from the user
            **kwargs: Additional context or parameters
            
        Returns:
            str: Agent's response
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class CustomerSupportAgent(BaseAgent):
    """
    Customer Support Agent implementation using direct Ollama calls with RAG.

    This agent retrieves relevant information from the knowledge base, builds
    a grounded support prompt, and sends it to a local Ollama model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self.knowledge_base = None
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.http_client = None
        self.http_module = None
        self.model = self.config.get("model", "llama3.1:8b")
        self.ollama_base_url = self.config.get("base_url", "http://localhost:11434").rstrip("/")
        self.temperature = float(self.config.get("temperature", 0.2))
        self.timeout = float(self.config.get("timeout", 60))
        self.max_input_chars = int(self.config.get("max_input_chars", 2000))
        self.max_context_docs = int(self.config.get("max_context_docs", 3))
        
    async def initialize(self) -> None:
        """
        TODO: Initialize the customer support agent.
        
        Steps:
        1. Initialize the LLM (e.g., OpenAI, Anthropic, local models)
        2. Set up the knowledge base/vector store
        3. Create RAG tool
        4. Create ReAct agent with tools
        5. Set up agent executor
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for Ollama integration. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        self.http_module = httpx
        self.http_client = httpx.AsyncClient(
            base_url=self.ollama_base_url,
            timeout=httpx.Timeout(self.timeout),
        )
        self.llm = {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.ollama_base_url,
        }

        if self.config.get("validate_connection", False):
            await self._check_ollama_connection()

        await self._setup_knowledge_base()
        tools = await self._create_tools()
        await self._create_agent(tools)

        self.is_initialized = True

    async def _check_ollama_connection(self) -> None:
        """Fail fast when explicitly asked to validate Ollama availability."""
        if not self.http_client:
            raise RuntimeError("Ollama HTTP client is not configured")

        try:
            response = await self.http_client.get("/api/tags")
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama and pull the configured model "
                f"with `ollama pull {self.model}`."
            ) from exc
    
    async def _setup_knowledge_base(self) -> None:
        """
        Set up the knowledge base for RAG using ChromaDB.
        
        This method automatically creates embeddings and stores them in ChromaDB.
        Students only need to implement the retrieval logic in _rag_search().
        """
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            import os
            import hashlib
            
            db_path = self.config.get("chroma_path", "./data/chroma_db")
            os.makedirs(db_path, exist_ok=True)

            self.embedding_model = SentenceTransformer(
                self.config.get("embedding_model", "all-MiniLM-L6-v2")
            )
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            collection_name = self.config.get("collection_name", "customer_support_kb")

            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                if self.collection.count() > 0:
                    logger.info("Knowledge base already exists with %s documents", self.collection.count())
                    return
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Customer support knowledge base"}
                )

            knowledge_documents = self._get_customer_support_documents()
            logger.info("Ingesting %s documents into knowledge base", len(knowledge_documents))
            
            documents = []
            metadatas = []
            ids = []
            
            for i, doc_data in enumerate(knowledge_documents):
                doc_id = f"doc_{i}_{hashlib.md5(doc_data['content'].encode()).hexdigest()[:8]}"
                
                documents.append(doc_data['content'])
                metadatas.append({
                    'category': doc_data['category'],
                    'title': doc_data['title'],
                    'doc_id': doc_id
                })
                ids.append(doc_id)

            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            logger.info("Successfully ingested %s documents into ChromaDB", len(documents))

        except Exception as e:
            logger.exception("Error setting up knowledge base")
            raise
    
    def _get_customer_support_documents(self) -> List[Dict[str, str]]:
        """
        Predefined customer support knowledge base.
        
        This is the definitive knowledge base that students will work with.
        Do not modify these documents - they form the complete knowledge base.
        """
        return [
            # Return Policy
            {
                "title": "Return Policy Overview",
                "category": "returns",
                "content": "We offer a 30-day return policy for all products purchased from our store. Items must be in original condition with all tags and packaging intact. Returns are processed within 5-7 business days of receiving the returned item. Refunds are issued to the original payment method."
            },
            {
                "title": "Return Process Steps",
                "category": "returns", 
                "content": "To initiate a return: 1) Log into your account and go to Order History, 2) Select the order and click 'Return Items', 3) Choose the items to return and reason, 4) Print the prepaid return label, 5) Pack items securely and attach the label, 6) Drop off at any UPS location or schedule pickup."
            },
            {
                "title": "Non-Returnable Items",
                "category": "returns",
                "content": "The following items cannot be returned: personalized or customized products, perishable goods, digital downloads, gift cards, intimate apparel, and items marked as final sale. Health and safety regulations prevent returns of opened cosmetics and personal care items."
            },
            
            # Shipping Information
            {
                "title": "Shipping Methods and Times",
                "category": "shipping",
                "content": "We offer multiple shipping options: Standard shipping (5-7 business days, free on orders over $50), Express shipping (2-3 business days, $12.99), Next-day shipping (1 business day, $24.99). All orders placed before 2 PM EST ship the same day."
            },
            {
                "title": "International Shipping",
                "category": "shipping",
                "content": "We ship internationally to over 50 countries. International shipping takes 7-14 business days via DHL Express. Shipping costs vary by destination and are calculated at checkout. Customers are responsible for customs fees and import duties. Some restrictions apply to certain products and countries."
            },
            {
                "title": "Order Tracking",
                "category": "shipping",
                "content": "Once your order ships, you'll receive a tracking number via email. Track your package using the tracking number on our website or the carrier's website. You can also track orders by logging into your account and viewing Order History. Tracking updates may take 24 hours to appear."
            },
            
            # Customer Support
            {
                "title": "Contact Information",
                "category": "support",
                "content": "Customer support is available 24/7 via multiple channels: Phone: 1-800-HELP-NOW (1-800-435-7669), Email: support@company.com, Live chat on our website (available 6 AM - 12 AM EST), or submit a support ticket through your account dashboard."
            },
            {
                "title": "Response Times",
                "category": "support",
                "content": "Our support team response times: Live chat - immediate during business hours, Phone support - average wait time under 3 minutes, Email support - response within 4 hours during business days, Support tickets - response within 24 hours. Premium customers receive priority support with faster response times."
            },
            
            # Warranty and Technical Support
            {
                "title": "Product Warranty",
                "category": "warranty",
                "content": "All products come with a manufacturer's warranty. Electronics have 1-year warranty covering defects and malfunctions. Apparel and accessories have 90-day warranty against material defects. Warranty claims require proof of purchase and must be initiated within the warranty period."
            },
            {
                "title": "Technical Support",
                "category": "technical",
                "content": "Free technical support is available for all electronic products. Our certified technicians provide assistance with setup, troubleshooting, and software issues. Technical support is available Monday-Friday 8 AM - 8 PM EST via phone or email. We also offer remote assistance for compatible devices."
            },
            
            # Account and Orders
            {
                "title": "Account Management",
                "category": "account",
                "content": "Manage your account online: Update personal information and addresses, view order history and tracking, manage payment methods, set communication preferences, download invoices and receipts. Account changes may take up to 24 hours to reflect across all systems."
            },
            {
                "title": "Order Modifications",
                "category": "orders",
                "content": "Orders can be modified or canceled within 1 hour of placement if not yet processed. Contact customer service immediately to make changes. Once an order is processed and shipped, it cannot be modified. You can return unwanted items following our return policy."
            },
            
            # Payment and Billing
            {
                "title": "Payment Methods",
                "category": "payment",
                "content": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, Google Pay, and Buy Now Pay Later options through Klarna and Afterpay. Gift cards and store credit can also be used for purchases. Payment is processed securely using 256-bit SSL encryption."
            },
            {
                "title": "Billing and Invoices",
                "category": "billing",
                "content": "Billing occurs when your order ships. You'll receive an email confirmation with invoice details. Invoices are available in your account under Order History. For business purchases, we can provide detailed invoices with tax information. Contact our billing department for any payment disputes or questions."
            },
            
            # Product Information
            {
                "title": "Product Availability",
                "category": "products",
                "content": "Product availability is updated in real-time on our website. If an item shows as 'In Stock', it's available for immediate shipping. 'Limited Stock' means fewer than 10 items remaining. 'Pre-order' items will ship on the specified release date. Out of stock items can be added to your wishlist for restock notifications."
            },
            {
                "title": "Size and Fit Guide",
                "category": "products",
                "content": "Each product page includes detailed size charts and fit information. For apparel, we recommend checking measurements against our size guide rather than relying on size labels from other brands. If you're between sizes, we generally recommend sizing up. Our customer service team can provide personalized fit recommendations."
            }
        ]
    
    async def _create_tools(self) -> List[Dict[str, Any]]:
        """
        Describe tools available to the direct RAG agent.
        
        Returns:
            List[Tool]: List of tools available to the agent
        """
        return [
            {
                "name": "knowledge_search",
                "description": "Search the customer support knowledge base for relevant information",
            }
        ]
    
    async def _rag_search(self, query: str) -> str:
        """
        Retrieve relevant knowledge base entries from ChromaDB.
        
        The knowledge base is already set up with documents and embeddings.
        You need to implement the search logic to retrieve relevant information.
        
        Args:
            query: Search query from the user (e.g., "What is your return policy?")
            
        Returns:
            str: Formatted relevant information from the knowledge base
            
        Available resources:
        - self.collection: ChromaDB collection with embedded documents
        - self.embedding_model: SentenceTransformer model for creating query embeddings
        
        Implementation steps:
        1. Use ChromaDB's query method to search for relevant documents
        2. Retrieve top 3-5 most relevant documents
        3. Format the results into a readable response
        4. Include document titles and categories for context
        
        ChromaDB query example:
        ```python
        results = self.collection.query(
            query_texts=[query],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        ```
        
        The results structure:
        - results['documents'][0]: List of document contents
        - results['metadatas'][0]: List of metadata (title, category)
        - results['distances'][0]: List of similarity distances (lower = more similar)
        """
        if not hasattr(self, 'collection') or self.collection is None:
            return "Knowledge base not available. Please ensure the service is properly initialized."

        query = (query or "").strip()
        if not query:
            return "Please provide a search query."

        try:
            n_results = int(self.config.get("rag_n_results", self.max_context_docs))
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, min(n_results, 5)),
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not documents:
                return "I couldn't find relevant information for that query."

            max_distance = self.config.get("rag_max_distance")
            formatted_results = []
            for doc, meta, distance in zip(documents, metadatas, distances):
                if max_distance is not None and distance > float(max_distance):
                    continue

                title = meta.get("title", "Knowledge Base Article")
                category = meta.get("category", "general")
                formatted_results.append(
                    f"{title} (category: {category}, relevance distance: {distance:.3f})\n{doc}"
                )

            if not formatted_results:
                return "I couldn't find sufficiently relevant information for that query."

            return "\n\n".join(formatted_results)

        except Exception as e:
            logger.exception("RAG search failed")
            return "Error searching knowledge base."
    
    async def _create_agent(self, tools: List[Dict[str, Any]]) -> None:
        """
        Create lightweight direct agent metadata.
        
        Args:
            tools: List of tools available to the agent
        """
        self.agent = {
            "type": "direct_ollama_rag",
            "tools": tools,
        }
    
    async def process_query(self, text: str, **kwargs) -> str:
        """
        Process user query using RAG context and Ollama.
        
        Args:
            text: User's query
            **kwargs: Additional context
            
        Returns:
            str: Agent's response
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")

        normalized_text = (text or "").strip()
        if not normalized_text:
            raise ValueError("Text query cannot be empty")
        if len(normalized_text) > self.max_input_chars:
            raise ValueError(f"Text query exceeds {self.max_input_chars} character limit")
        if not self.http_client:
            raise RuntimeError("LLM client not initialized")

        context = await self._rag_search(normalized_text)
        prompt = self._build_support_prompt(normalized_text, context)

        try:
            response = await self.http_client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.exception("Ollama request failed")
            raise RuntimeError(
                "LLM service is unavailable. Ensure Ollama is running and the configured model is pulled."
            ) from exc

        answer = (payload.get("response") or "").strip()
        if not answer:
            raise RuntimeError("LLM returned an empty response")
        return answer

    def _build_support_prompt(self, user_query: str, context: str) -> str:
        """Build a compact grounded prompt for the local LLM."""
        return (
            "You are a professional customer support assistant.\n"
            "Answer using only the knowledge base context below. If the context does not contain "
            "the answer, say you do not have enough information and suggest contacting support. "
            "Be concise, friendly, and avoid inventing policies.\n\n"
            f"Knowledge base context:\n{context}\n\n"
            f"Customer question:\n{user_query}\n\n"
            "Answer:"
        )
    
    async def cleanup(self) -> None:
        """
        Cleanup agent resources.
        """
        if self.http_client:
            await self.http_client.aclose()
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self.http_client = None
        self.is_initialized = False
