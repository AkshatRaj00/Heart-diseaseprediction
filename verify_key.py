from groq import Groq

# Screen par jo tumhari key thi:
REAL_KEY = "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"

with open(".env", "w", encoding="utf-8") as f:
    f.write(f"GROQ_API_KEY={REAL_KEY}\n")

print("Saved key to .env successfully.")

client = Groq(api_key=REAL_KEY)
res = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Respond with 5 words: Confirm system is operational."}]
)
print("Groq Live Output:", res.choices[0].message.content)
