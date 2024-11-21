import os
import json
import logging
from uuid import uuid4
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader

from config.settings import MEMORIES_DIR, DATE_FORMAT

class DocumentManager:
    logger: logging.Logger
    memory_dir: str

    def __init__(self, memory_dir: str = MEMORIES_DIR) -> None:
        self.logger = logging.getLogger(__name__)
        self.memory_dir = memory_dir

    def save_document(self, memory_type: str, document: Document) -> None:
        """
        Save a document to the memory directory.
        """
        # Set the path and document structure
        timestamp: str = datetime.now().strftime(DATE_FORMAT)
        document_relevance_score: int = document.metadata.get("relevance_score", 5)
        directory: str = os.path.join(self.memory_dir, memory_type)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save the document metadata
        document.id = str(uuid4())
        document.metadata["timestamp"] = timestamp
        document.metadata["relevance_score"] = document_relevance_score

        # Set the file path
        file_path: str = os.path.join(directory, f"{document.id}_{timestamp}.json")

        # Save the document as JSON
        with open(file_path, "w") as f:
            json.dump({"page_content": document.page_content, "metadata": document.metadata}, f)
        self.logger.info(f"{memory_type.capitalize()} memory saved to {file_path}.")
        return document

    def save_document_from_text(self, memory_type: str, text: str, relevance_score: int = 5) -> None:
        """
        Save a document to the memory directory from text.
        """
        # Create a document from the text
        document = Document(page_content=text, metadata={"source": memory_type, "relevance_score": relevance_score})
        return self.save_document(memory_type, document)

    def load_documents(self, memory_type: str) -> list[Document]:
        """
        Load documents from the memory directory.
        """
        # Start with no documents loaded
        loaded_documents: list[Document] = []

        # Set the path and document structure
        dir_path: str = os.path.join(self.memory_dir, memory_type)
        jq_schema = '{page_content: .page_content, metadata: .metadata}'

        # Check if the directory exists, if not return empty list
        if not os.path.exists(dir_path):
            self.logger.warning(f"Directory {dir_path} does not exist.")
            return loaded_documents

        for filename in os.listdir(dir_path):
            # Load the documents from the directory
            filepath: str = os.path.join(dir_path, filename)

            # But only if the file is a JSON file
            if filename.endswith(".json"):
                try:
                    # Load the documents from the JSON file
                    retriever = JSONLoader(file_path=filepath, jq_schema=jq_schema, text_content=False)
                    retrieved_docs = retriever.load()

                    # Append the loaded documents to the list
                    for doc in retrieved_docs:
                        # Convert non-string content to JSON string
                        if not isinstance(doc.page_content, str):
                            doc.page_content = json.dumps(doc.page_content)
                        loaded_documents.append(doc)

                except json.JSONDecodeError:
                    self.logger.error(f"Failed to decode JSON for file {filename}. Skipping.")

                except Exception as e:
                    self.logger.error(f"Error loading file {filename}: {e}")
         
        return loaded_documents