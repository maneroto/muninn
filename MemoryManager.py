import logging
from uuid import uuid4
from langchain_nomic import NomicEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MemoryManager:
    logger: logging.Logger
    knowledge_vectors: dict
    embedding_model: NomicEmbeddings
    text_splitter: RecursiveCharacterTextSplitter

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
        self.knowledge_vectors = {
            "episodic": {
                "store": SKLearnVectorStore(embedding=self.embedding_model),
                "count": 0
            },
            "working": {
                "store": SKLearnVectorStore(embedding=self.embedding_model),
                "count": 0
            },
            "external": {
                "store": SKLearnVectorStore(embedding=self.embedding_model),
                "count": 0
            }
        }

    def __split_documents(self, documents: list[Document]):
        doc_splits = self.text_splitter.split_documents(documents)
        texts = [doc.page_content for doc in doc_splits]
        metadatas = [doc.metadata for doc in doc_splits]
        return texts, metadatas
    
    def set_knowledge_vector_from_documents(self, memory_type: str, documents: list[Document]):
        if not documents:
            self.logger.warning(f"No documents provided for {memory_type} memory vector store.")
            return
        
        # Split documents into smaller chunks   
        texts, metadatas = self.__split_documents(documents)

        # Populate vector store with embedded texts
        self.knowledge_vectors[memory_type]["store"] = SKLearnVectorStore.from_texts(texts, embedding=self.embedding_model, metadatas=metadatas, ids=[uuid4() for _ in range(len(texts))])
        self.knowledge_vectors[memory_type]["count"] = len(texts)

        self.logger.info(f"Initialized {memory_type} memory vector store with {len(texts)} documents.")

    def set_knowledge_vectors_from_documents(self, episodic_documents: list[Document], working_documents: list[Document], external_documents: list[Document]):
        self.set_knowledge_vector_from_documents("episodic", episodic_documents)
        self.set_knowledge_vector_from_documents("working", working_documents)
        self.set_knowledge_vector_from_documents("external", external_documents)

    def create_memory(self, memory_type: str, document: Document):
        texts, metadatas = self.__split_documents([document])
        vector_store: SKLearnVectorStore = self.knowledge_vectors[memory_type]["store"]
        vector_store.add_texts(texts, metadatas=metadatas, ids=[uuid4() for _ in range(len(texts))])
        self.knowledge_vectors[memory_type]["count"] += len(texts)

    def update_memory(self, memory_type: str, memory_id: str, new_memory: str):
        vector_store: SKLearnVectorStore = self.knowledge_vectors[memory_type]["store"]
        vector_store.add_texts([new_memory], kwargs={"id": memory_id})

    def read_memory(self, memory_type: str = "episodic", query: str = None, k: int=3):
        vector_store: SKLearnVectorStore = self.knowledge_vectors[memory_type]["store"]

        # Ensure query is a single string
        if not query:
            self.logger.warning("No query provided. Returning None.")
            return None
        
        if isinstance(query, list):
            self.logger.warning("Query is a list. Using the first element as the query.")
            query = query[0]
        elif not isinstance(query, str):
            self.logger.warning("Query is not a string. Converting to string.")
            query = str(query)

        # Check the number of available documents in the vector store
        if self.knowledge_vectors[memory_type]["count"] < 4:
            self.logger.warning(f"{memory_type.capitalize()} memory vector store is not enough to retrieve memories.")
            return None

        # Dynamically adjust `k` to avoid exceeding the number of documents
        adjusted_k: int = min(k, self.knowledge_vectors[memory_type]["count"])

        return vector_store.as_retriever(k=adjusted_k, kwargs= { 
                "search_type": "similarity_score_threshold",
                "score_threshold": 0.9
            }
        ).invoke(query)
    
    def reinforce_memory(self, memory_type: str, document: Document):
        vector_store: SKLearnVectorStore = self.knowledge_vectors[memory_type]["store"]
        document_id = document.metadata.get("id")
        if document_id:
            document.metadata["relevance_score"] = document.metadata.get("relevance_score", 0) + 1
            vector_store.add_documents([document], replace=True)
            self.logger.info(f"Reinforced {memory_type} memory with updated relevance score.")