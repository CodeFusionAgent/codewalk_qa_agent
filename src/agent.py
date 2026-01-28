from dotenv import load_dotenv
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message
from google import genai

from messenger import Messenger

load_dotenv()


class Agent:
    def __init__(self):
        
        self.messenger = Messenger()
        self.client = genai.Client()
        # Initialize other state here

    async def run(self, message: Message, updater: TaskUpdater) -> None:

        input_text = get_message_text(message)

        print(f"Input text: {input_text}")

        # Replace this example code with your agent logic

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking on the input...")
        )
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a debater on a given topic. You will be given a topic and you need to argue for or against it.",
            ),
            contents=input_text,
        )
        await updater.add_artifact(
            parts=[Part(root=DataPart(data={"argument": response.text}, kind="data"))],
            name="Response",
        )
