import logging

from config.settings import PROMPT_PATHS

class PromptLoader:
    logger: logging.Logger
    prompts: dict

    def __init__(self, prompt_paths: dict = PROMPT_PATHS):
        self.logger = logging.getLogger(__name__)
        self.prompts = {name: self._load_prompt(path) for name, path in prompt_paths.items()}

    def _load_prompt(self, filepath):
        """
        Load a prompt from a file.
        """
        with open(filepath, 'r') as f:
            self.logger.info(f"Loaded prompt from {filepath}.")
            return f.read()

    def get_goal_prompt(self, user_input) -> str:
        """
        Get the goal retrieval prompt.
        """
        system_message = self.prompts["goal_retrieval"].format(input=user_input)
        human_message = "Please generate a response based on the above information."
        return system_message, human_message
    
    def get_task_segmentation_prompt(self, goal_retrieval: dict) -> str:
        """
        Get the task segmentation prompt.
        """
        system_message = self.prompts["task_segmentation"].format(
            input=goal_retrieval['input'],
            goal=goal_retrieval['goal'],
            obstacle=goal_retrieval['obstacle'],
            relevant_information=", ".join(goal_retrieval['analysis']['relevant_information'])
        )
        human_message = "Please generate a response based on the above information."
        return system_message, human_message
    
    def get_response_prompt(self, goal_retrieval: dict, retrieved_data: dict, conversation_context: list[str]) -> tuple[str, str]:
        """
        Get the response generation prompt.
        """
        system_message = self.prompts["response_generation"].format(
            goal=goal_retrieval['goal'],
            obstacle=goal_retrieval['obstacle'],
            relevant_information=", ".join(goal_retrieval['analysis']['relevant_information']),
            episodic_memory=retrieved_data['recall_episodic_memory'],
            external_source=retrieved_data['external_source'],
            ask_information=retrieved_data['ask_information'],
            conversation_context=", ".join(conversation_context)
        )
        human_message = "Please generate a response based on the above information."
        return system_message, human_message
    
    def get_memory_synthesizer_prompt(self, user_input: str, goal_retrieval: dict, retrieved_data: dict) -> tuple[str, str]:
        """
        Synthesize memory content.
        """
        system_message = self.prompts["memory_synthesizer"].format(
            input=user_input,
            goal=goal_retrieval['goal'],
            obstacle=goal_retrieval['obstacle'],
            relevant_information=", ".join(goal_retrieval['analysis']['relevant_information']),
            episodic_memory=retrieved_data['recall_episodic_memory'],
            external_source=retrieved_data['external_source'],
            ask_information=retrieved_data['ask_information']
        )
        human_message = "Please generate a response based on the above information."
        return system_message, human_message