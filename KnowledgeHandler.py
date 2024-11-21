import json
import logging
from uuid import uuid4
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from DocumentManager import DocumentManager
from MemoryManager import MemoryManager
from MemoryMaintenance import MemoryMaintenance

class KnowledgeHandler:
    logger: logging.Logger
    document_manager: DocumentManager
    memory_manager: MemoryManager
    memory_mainenance: MemoryMaintenance

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.document_manager = DocumentManager()
        self.memory_manager = MemoryManager()
        # self.memory_mainenance = MemoryMaintenance()

        self.memory_manager.set_knowledge_vectors_from_documents(
            self.document_manager.load_documents(memory_type="episodic"),
            self.document_manager.load_documents(memory_type="working"),
            self.document_manager.load_documents(memory_type="external"),
        )

    def create_memory(self, memory_type: str, text: str):
        memory_document = self.document_manager.save_document_from_text(memory_type, text)
        self.memory_manager.create_memory(memory_type, memory_document)

    # def find_memories(self, memory_type: str, queries: list[str]):
    #     memories: list[Document] = []
    #     for query in queries:
    #         found_memories = self.memory_manager.read_memory(memory_type, query)
    #         if found_memories:
    #             for memory in found_memories:
    #                 page_content = memory.page_content
    #                 json_data = json.loads(page_content)
    #                 memories.append(json_data['page_content'])
    #     return memories

    def find_memories(self, memory_type: str, queries: list[str]):
        memories: list[str] = []
        for query in queries:
            found_memories = self.memory_manager.read_memory(memory_type, query)
            if found_memories:
                for memory in found_memories:
                    page_content = memory.page_content

                    # Check if page_content is in JSON format (starts with '{' or '[')
                    if isinstance(page_content, str) and (page_content.startswith("{") or page_content.startswith("[")):
                        try:
                            json_data = json.loads(page_content)
                            # Ensure 'page_content' exists within the JSON data
                            memories.append(json_data.get('page_content', page_content))
                        except json.JSONDecodeError:
                            # If JSON parsing fails, add raw page_content
                            memories.append(page_content)
                    else:
                        # Add page_content directly if not in JSON format
                        memories.append(page_content)
        
        return memories

    def find_memory(self, memory_type: str, query: str):
        return self.memory_manager.read_memory(memory_type, query)
    
    def update_memory(self, memory_type: str, text: str):
        self.memory_manager.update_memory(memory_type, text)

    def perform_external_search(self, query: str, k: int=3):
        web_search_tool = TavilySearchResults(k=3)

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

        self.logger.info(f"Performing external search for query: {query}")

        web_documents = web_search_tool.invoke({
            "query": query,
        })
        web_results = [Document(page_content=doc["content"], metadata={"source": "web"}, id=uuid4()) for doc in web_documents]

        self.memory_manager.set_knowledge_vector_from_documents("external", web_results)