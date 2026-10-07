import os
import sys
from groq import Groq

print("="*60)
key = input("Groq Console se copy ki hui nayi key yahan paste karo: ").strip().strip('"').strip("'")
print("="*60)

if not key.startswith("gsk_"):
    print("[ERROR] Key 'gsk_' se shuru honi chahiye. Tumne galat text paste kiya hai.")
    sys.exit(1)

# Test the key directly against Groq API
print(f"Testing key: {key[:10]}... against Groq LLaMA-3.3-70B")
try:
    client = Groq(api_key=key)
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Respond: 'API Key Verified Successfully.'"}]
    )
    print("\n[SUCCESS! GROQ LIVE RESPONSE]:", res.choices[0].message.content)
    
    # Write to .env permanently
    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"GROQ_API_KEY={key}\n")
    print("[PERMANENT] .env file successfully locked with active key!\n")

except Exception as e:
    print(f"\n[FAIL] Groq rejected this key with: {e}")
    print("Kripya console.groq.com/keys par check karein ki account active hai ya nahi.")
