import os
import json
import lancedb
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from lancedb.pydantic import LanceModel, Vector
import pyarrow as pa


class TheArchives:
    """
    Retrieval-Augmented Generation (RAG) system using LanceDB and Google's Gemini.
    
    TheArchives provides vector-based semantic search over a knowledge base,
    using Google's embedding models for vectorization and Gemini for generation.
    Supports ingestion from JSON, single document addition, and RAG-based queries.
    
    Attributes:
        db_path: Path to LanceDB database directory
        table_name: Name of the table storing knowledge entries
        embedding_model: Google embedding model name for vectorization
        embedding_dim: Dimension of embedding vectors (768 for embedding-001)
        generation_model_name: Gemini model name for answer generation
        db: LanceDB connection instance
        generator_model: Configured Gemini model for text generation
    """
    
    def __init__(
        self, 
        google_api_key: str,
        db_path: str = "./lancedb_data",
        table_name: str = "general_knowledge",
        embedding_model: str = "models/embedding-001",
        generation_model: str = "gemini-1.5-flash",
        embedding_dim: int = 768
    ):
        """
        Initialize TheArchives RAG system.
        
        Args:
            google_api_key: Google API key for Gemini and embedding access.
                           Get from https://makersuite.google.com/app/apikey
            db_path: Directory path for LanceDB storage. Created if doesn't exist.
            table_name: Table name for storing knowledge entries.
            embedding_model: Google embedding model identifier.
                           Default: "models/embedding-001" (768 dimensions)
            generation_model: Gemini model for answer generation.
                            Default: "gemini-1.5-flash" for speed
            embedding_dim: Embedding vector dimensionality. Must match model.
        

        """
        # Configure GenAI
        genai.configure(api_key=google_api_key)
        
        self.db_path: str = db_path
        self.table_name: str = table_name
        self.embedding_model: str = embedding_model
        self.embedding_dim: int = embedding_dim
        self.generation_model_name: str = generation_model
        
        # Connect to DB
        self.db: lancedb.DBConnection = lancedb.connect(self.db_path)
        self._init_tables()

        # Initialize Generator Model
        self.generator_model: genai.GenerativeModel = genai.GenerativeModel(
            self.generation_model_name
        )

    def _init_tables(self) -> None:
        """
        Initialize database tables without schema constraints.
        
        Tables are created dynamically on first data insert to allow flexible
        schema evolution. This method checks if table exists and logs status.
        
        Note:
            Table schema is inferred from first batch of data inserted.
        """
        if self.table_name not in self.db.table_names():
            print(f"Table '{self.table_name}' will be created on first data insert")

    def _generate_embedding(
        self, 
        text: str, 
        task_type: str = "retrieval_document"
    ) -> List[float]:
        """
        Generate embedding vector for a single text using Google's embedding API.
        
        Args:
            text: Text content to embed. Can be document, query, or any string.
            task_type: Task type hint for the embedding model. Options:
                      - "retrieval_document": For documents being indexed
                      - "retrieval_query": For search queries
                      - "semantic_similarity": For similarity comparisons
                      - "classification": For categorization tasks
        
        Returns:
            List[float]: Embedding vector of length embedding_dim (default 768).

        
       """
        result: Dict[str, Any] = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type=task_type,
        )
        return result["embedding"]

    def _ensure_index(self) -> None:
        """
        Create IVF_PQ index on text_vector column if sufficient data exists.
        
        IVF_PQ (Inverted File with Product Quantization) is a fast approximate
        nearest neighbor search index. Requires minimum 256 rows for creation.
        
        Index creation improves search performance on large datasets but adds
        overhead for small datasets, hence the row count check.
        
        """
        table: lancedb.Table = self.db.open_table(self.table_name)
        row_count: int = table.count_rows()
        
        if row_count == 0:
            return
        
        if row_count < 256:
            print(
                f"Table has {row_count} rows. IVF_PQ requires at least 256 rows. "
                f"Using flat index for now."
            )
            return
        
        try:
            # Check if index already exists
            indices: List = table.list_indices()
            if any('text_vector' in str(idx) for idx in indices):
                print("Index already exists")
                return
                
            table.create_index(
                vector_column_name="text_vector",
                index_type="IVF_PQ",
            )
            print(f"Created IVF_PQ index on text_vector")
        except Exception as e:
            print(f"Note: Could not create index: {e}")

    def ingest_json(self, file_path: str) -> None:
        """
        Parse and bulk insert JSON data into LanceDB.
        
        Expected JSON format:
        [
            {
                "id": "mem_001",
                "content": "Knowledge content here",
                "tags": ["tag1", "tag2"],
                "meta": {"source": "book", "date": "2024"}
            },
            ...
        ]
        
        Args:
            file_path: Path to JSON file containing knowledge entries.
                      Must be valid JSON array of objects.
        
        Returns:
            None. Prints progress during ingestion.
        
        """
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return

        print(f"Loading data from {file_path}...")
        
        with open(file_path, 'r') as f:
            raw_data: List[Dict[str, Any]] = json.load(f)

        print(f"Generating embeddings for {len(raw_data)} entries...")
        records_to_add: List[Dict[str, Any]] = []
        
        for i, item in enumerate(raw_data):
            text_content: str = item.get("content", "")
            
            # Generate embedding
            embedding: List[float] = self._generate_embedding(text_content)
            
            record: Dict[str, Any] = {
                "text": text_content,
                "text_vector": embedding,
                "memory_id": item.get("id", "unknown_id"),
                "tags": json.dumps(item.get("tags", [])), 
                "meta_data": json.dumps(item.get("meta", {}))
            }
            records_to_add.append(record)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(raw_data)} entries...")

        # Create or open table
        if self.table_name in self.db.table_names():
            table: lancedb.Table = self.db.open_table(self.table_name)
            table.add(records_to_add)
        else:
            table = self.db.create_table(self.table_name, data=records_to_add)
        
        print(
            f"Successfully loaded {len(records_to_add)} entries "
            f"into {self.table_name}."
        )
        
        # Try to create index if enough data
        self._ensure_index()

    def add_knowledge(
        self, 
        content: str, 
        memory_id: str, 
        tags: List[str], 
        metadata: Dict[str, Any] = {}
    ) -> None:
        """
        Add a single knowledge entry to the database.
        
        Args:
            content: Text content of the knowledge entry. The main searchable text.
            memory_id: Unique identifier for this entry (e.g., "mem_001", "doc_42").
            tags: List of category tags for filtering/organization (e.g., ["ai", "matrix"]).
            metadata: Additional structured data about the entry (e.g., source, date, author).
                     Default: empty dict.
        
        Returns:
            None. Prints confirmation message with memory_id.
        
        """
        # Generate embedding
        embedding: List[float] = self._generate_embedding(content)
        
        record: Dict[str, Any] = {
            "text": content,
            "text_vector": embedding,
            "memory_id": memory_id,
            "tags": json.dumps(tags),
            "meta_data": json.dumps(metadata)
        }
        
        if self.table_name in self.db.table_names():
            table: lancedb.Table = self.db.open_table(self.table_name)
            table.add([record])
        else:
            self.db.create_table(self.table_name, data=[record])
            
        print(f"Stored: {memory_id}")

    def search_knowledge(
        self, 
        query: str, 
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant knowledge using semantic vector similarity.
        
        Generates an embedding for the query and finds the most similar
        documents in the knowledge base using cosine similarity.
        
        Args:
            query: Natural language search query (e.g., "What is the Matrix?").
            limit: Maximum number of results to return. Default: 3.
                  Range: 1-100 recommended.
        
        Returns:
            List[Dict[str, Any]]: List of matching entries, each containing:
                - memory_id (str): Unique entry identifier
                - content (str): Full text content
                - tags (List[str]): Associated tags
                - metadata (Dict): Additional metadata
                - distance (float): Vector distance (lower = more similar)
            
            Empty list if no results found or table doesn't exist.
        
        """
        try:
            if self.table_name not in self.db.table_names():
                print("Warning: Table does not exist")
                return []
                
            table: lancedb.Table = self.db.open_table(self.table_name)
            
            if table.count_rows() == 0:
                print("Warning: Table is empty")
                return []
            
            # Generate query embedding
            query_embedding: List[float] = self._generate_embedding(
                query, 
                task_type="retrieval_query"
            )
            
            # Search using the embedding
            results: List[Dict[str, Any]] = (
                table.search(query_embedding, vector_column_name="text_vector")
                .limit(limit)
                .to_list()
            )
            
            clean_results: List[Dict[str, Any]] = []
            for r in results:
                clean_results.append({
                    "memory_id": r.get("memory_id", ""),
                    "content": r["text"],
                    "tags": json.loads(r["tags"]),
                    "metadata": json.loads(r["meta_data"]),
                    "distance": r.get("_distance", 0.0)
                })
            return clean_results
            
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            import traceback
            traceback.print_exc()
            return []

    @classmethod
    def from_env(
        cls, 
        db_path: str, 
        api_key: Optional[str] = None
    ) -> "TheArchives":
        """
        Factory method to create TheArchives instance using environment variables.
        
        Convenience method that reads GOOGLE_API_KEY from environment if not
        provided explicitly. Useful for production deployments.
        
        Args:
            db_path: Path to LanceDB database directory.
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
        
        Returns:
            TheArchives: Configured instance ready for use.
        
        """
        return cls(
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            db_path=db_path,
            table_name="general_knowledge"
        )

    def answer_query(self, user_query: str) -> str:
        """
        Answer a query using Retrieval-Augmented Generation (RAG).
        
        Process:
        1. Search knowledge base for relevant context
        2. Construct prompt with context and query
        3. Generate answer using Gemini
        
        Args:
            user_query: Natural language question to answer
                       (e.g., "What is the nature of reality?").
        
        Returns:
            str: Generated answer based on retrieved knowledge.
                Returns "I have no relevant memories regarding that."
                if no relevant context found.
        
        """
        context: List[Dict[str, Any]] = self.search_knowledge(user_query)
        
        if not context:
            return "I have no relevant memories regarding that."
        
        context_str: str = "\n\n".join([
            f"[{c['memory_id']}]: {c['content']}" 
            for c in context
        ])
        
        prompt: str = f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {user_query}

Answer:"""
        
        response = self.generator_model.generate_content(prompt)
        return response.text