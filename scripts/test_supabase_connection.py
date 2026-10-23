import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SECRET_KEY")

if not url or not key:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_SECRET_KEY in .env"
    )

supabase = create_client(url, key)

response = (
    supabase.table("document_chunks")
    .select("id")
    .limit(1)
    .execute()
)

print("Supabase connection successful!")
print("Rows returned:", len(response.data))