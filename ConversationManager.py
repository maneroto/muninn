import collections
from langchain_core.documents import Document

class ConversationManager:
    def __init__(self, max_context_size=5):
        # Store recent messages for conversational context
        self.conversation_history = collections.deque(maxlen=max_context_size)

    def add_message(self, message):
        """Add a message to the conversation history."""
        self.conversation_history.append(message)

    def get_context(self):
        """Retrieve the current conversation context."""
        return list(self.conversation_history)

    def clear_context(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
