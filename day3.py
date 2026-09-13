import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
text = "AI Engineer Track 2: LLM Fundamentals"

tokens = enc.encode(text)

print(f"Text: {text}")
print(f"Tokens IDs: {tokens}")
print(f"Token count: {len(tokens)}")
print("\nDecoded back:")
for t in tokens:
    print(f" {t} -> '{enc.decode([t])}'")

# Try your name!
my_text = "Hello, I am Ahmed learning AI Engineering"
print(f"\nYour test: '{my_text}' = {len(enc.encode(my_text))} tokens")
