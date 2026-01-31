import logging
import os
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codewalk_qa")

QA_DATA_DIR = os.getenv("QA_DATA_DIR", "data")
QA_MODEL = os.getenv("QA_MODEL", "gemini-2.5-flash")
QA_API_KEY = os.getenv("QA_API_KEY")

# Model determines base_url automatically
LLM_MODELS = {
    "gpt-4o": None,
    "gpt-4o-mini": None,
    "claude-sonnet-4-5": "https://api.anthropic.com/v1/",
    "gemini-2.5-flash": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "gemini-2.0-flash": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


def load_qa_data(data_dir: str) -> list[dict]:
    """Load all Q&A YAML files from the data directory."""
    qa_data = []
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return qa_data
    for yaml_file in data_path.glob("*.yaml"):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                if data and "questions" in data:
                    num_questions = len(data["questions"])
                    qa_data.extend(data["questions"])
                    logger.info(f"Loaded {num_questions} questions from {yaml_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {yaml_file.name}: {e}")
    logger.info(f"Total questions loaded: {len(qa_data)}")
    return qa_data


def find_answer(question: str, qa_data: list[dict]) -> tuple[str | None, str | None]:
    """Search for a matching question in the Q&A data. Returns (answer, key_name)."""
    question_lower = question.lower().strip()
    for qa in qa_data:
        if qa.get("question", "").lower().strip() == question_lower:
            for key, value in qa.items():
                if key.endswith("_answer") and key != "reference_answer" and value:
                    return value, key
    return None, None


class Agent:
    def __init__(self):
        self.qa_data = load_qa_data(QA_DATA_DIR)
        self.model = QA_MODEL
        base_url = LLM_MODELS.get(self.model, LLM_MODELS["gemini-2.5-flash"])
        self.client = openai.OpenAI(api_key=QA_API_KEY, base_url=base_url)

    def generate_answer(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior software engineer. Answer questions about codebases accurately."},
                {"role": "user", "content": question}
            ],
        )
        return response.choices[0].message.content.strip()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        question = get_message_text(message)
        logger.info(f"Received question: {question[:100]}...")
        answer, key_name = find_answer(question, self.qa_data)
        if answer:
            logger.info(f"Found answer from YAML using key: {key_name}")
        else:
            logger.info(f"No YAML match found, generating answer with {self.model}")
            answer = self.generate_answer(question)
            logger.info("Answer generated successfully")
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=answer))],
            name="Answer",
        )
