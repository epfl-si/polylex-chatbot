from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"

def load_project_env():
    load_dotenv(dotenv_path=ENV_PATH)
    return ENV_PATH
