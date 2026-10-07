import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
key = os.getenv("GROQ_API_KEY", "").strip()
print("Using Key:", key[:12] + "...")

client = Groq(api_key=key)
res = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Respond with 5 words: Confirm system is operational."}]
)
print("Groq Response:", res.choices[0].message.content)
