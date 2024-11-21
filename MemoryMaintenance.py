from datetime import datetime, timedelta
from langchain_core.documents import Document
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMaintenance:
    def __init__(self, vector_manager):
        self.vector_manager = vector_manager

    def decay_memory(self, memory_type="episodic", decay_threshold=1):
        vector_store = getattr(self.vector_manager, f"{memory_type}_memory_vector")
        all_documents = vector_store.get_all_documents()
        for doc in all_documents:
            relevance_score = doc.metadata.get("relevance_score", 0)
            if relevance_score <= decay_threshold:
                vector_store.remove_documents([doc])
                logger.info(f"Decayed {memory_type} memory due to low relevance.")
            else:
                doc.metadata["relevance_score"] -= 1
                vector_store.add_documents([doc], replace=True)

    def summarize_memories(self, memory_type="episodic", max_age_days=30, max_memories=5):
        vector_store = getattr(self.vector_manager, f"{memory_type}_memory_vector")
        age_threshold = datetime.now() - timedelta(days=max_age_days)
        all_documents = vector_store.get_all_documents()
        older_memories = [doc for doc in all_documents if datetime.fromisoformat(doc.metadata["timestamp"]) < age_threshold]

        if len(older_memories) > max_memories:
            selected_memories = older_memories[:max_memories]
            summarized_content = " ".join(doc.page_content for doc in selected_memories)
            summary_text = f"Summary of memories: {summarized_content[:500]}..."
            summary_document = Document(page_content=summary_text)
            self.vector_manager.save_document(summary_document, memory_type)
            for doc in selected_memories:
                vector_store.remove_documents([doc])
            logger.info(f"Summarized older entries in {memory_type} memory.")
