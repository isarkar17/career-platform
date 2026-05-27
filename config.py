import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
SUPABASE_URL      = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY      = os.environ.get("SUPABASE_KEY", "")
RAG_API_KEY       = os.environ.get("RAG_API_KEY", "change-this-in-secrets")
SECRET_KEY        = os.environ.get("SECRET_KEY", os.environ.get("SESSION_SECRET", "change-this-flask-secret-key"))

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "lenny_archive"
