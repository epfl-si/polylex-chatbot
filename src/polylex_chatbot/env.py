from pathlib import Path
from dotenv import load_dotenv

def load_project_env(env_path=None):
    if env_path is None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path)
    return env_path
