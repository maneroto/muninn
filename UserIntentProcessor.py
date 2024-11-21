import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from PromptLoader import PromptLoader
from KnowledgeHandler import KnowledgeHandler
from ConversationManager import ConversationManager
from config.settings import MODEL_NAME, MODEL_TEMPERATURE, MODEL_FORMAT

class UserIntentProcessor:
    logger: logging.Logger
    prompt_loader: PromptLoader
    knowledge_handler: KnowledgeHandler
    ai_model: ChatOllama

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prompt_loader = PromptLoader()
        self.knowledge_handler = KnowledgeHandler()
        self.conversation_manager = ConversationManager()
        self.ai_model = ChatOllama(model=MODEL_NAME, temperature=MODEL_TEMPERATURE, format=MODEL_FORMAT)

    def retrieve_goal(self, user_input: str):
        system_message, human_message = self.prompt_loader.get_goal_prompt(user_input)
        goal_retrieval = self.ai_model.invoke(
            [SystemMessage(content=system_message)]
            + [HumanMessage(content=human_message)]
        )
        print(goal_retrieval)
        goal_retrieval = json.loads(goal_retrieval.content)
        self.logger.info(f"Goal retrieval response: {goal_retrieval}")
        return goal_retrieval
    
    def segment_tasks(self, goal_retrieval: dict):
        system_message, human_message = self.prompt_loader.get_task_segmentation_prompt(goal_retrieval)
        task_segmentation = self.ai_model.invoke(
            [SystemMessage(content=system_message)]
            + [HumanMessage(content=human_message)]
        )
        task_segmentation = json.loads(task_segmentation.content)
        self.logger.info(f"Task segmentation response: {task_segmentation}")
        return task_segmentation
    
    def generate_response(self, goal_retrieval: dict, retrieved_data: dict):
        system_message, human_message= self.prompt_loader.get_response_prompt(goal_retrieval, retrieved_data, self.conversation_manager.get_context())
        response = self.ai_model.invoke(
            [SystemMessage(content=system_message)]
            + [HumanMessage(content=human_message)]
        )
        response = json.loads(response.content)
        self.logger.info(f"Response generation response: {response}")
        return response
    
    def synthesize_memory(self, user_input: str, goal_retrieval: dict, retrieved_data: dict):
        system_message, human_message= self.prompt_loader.get_memory_synthesizer_prompt(user_input, goal_retrieval, retrieved_data)
        synthesized_memory = self.ai_model.invoke(
            [SystemMessage(content=system_message)]
            + [HumanMessage(content=human_message)]
        )
        synthesized_memory = json.loads(synthesized_memory.content)
        self.logger.info(f"Memory synthesizer response: {synthesized_memory}")
        return synthesized_memory
    
    def retrieve_data(self, task_segmentation: dict):
        external_source: list[str] = task_segmentation["retrieval_decision"]["external_source"]
        recall_episodic_memory: list[str] = task_segmentation["retrieval_decision"]["recall_episodic_memory"]
        ask_information: list[str] = task_segmentation["retrieval_decision"]["ask_information"]

        # if(len(external_source) != 0):
        #     for source in external_source:
        #         self.knowledge_handler.perform_external_search(source)
            
        return {
            "external_source": self.knowledge_handler.find_memories("external", external_source),
            "recall_episodic_memory": self.knowledge_handler.find_memories("episodic", recall_episodic_memory),
            "ask_information": ask_information
        }
    
    def process_user_intent(self, user_input: str):
        # Add the user input to the conversation context
        self.conversation_manager.add_message(user_input)

        goal_retrieval = self.retrieve_goal(user_input)
        task_segmentation = self.segment_tasks(goal_retrieval)
        retrieved_data = self.retrieve_data(task_segmentation)
        response = self.generate_response(goal_retrieval, retrieved_data)
        print(response)
        response = response['response']

        self.conversation_manager.add_message(response)

        memory = self.synthesize_memory(user_input, goal_retrieval, retrieved_data)
        self.knowledge_handler.create_memory("episodic", memory['memory'])
        return response