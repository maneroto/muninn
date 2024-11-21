import logging
from dotenv import load_dotenv

from UserIntentProcessor import UserIntentProcessor

class App:
    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.user_intent_processor = UserIntentProcessor()

    def run(self):
        while True:
            user_input = input("Enter your query (or type 'exit' to quit): ")

            if user_input == "exit":
                break

            response = self.user_intent_processor.process_user_intent(user_input)

            print(response)

if __name__ == "__main__":
    load_dotenv()
    app = App()
    app.run()