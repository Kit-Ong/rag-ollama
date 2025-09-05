import ollama

response = ollama.chat(
    model='gemma3:12b', 
    messages=[{
        'role': 'user', 
        'content': 'This is an image about flow diagram of a banking app. Can you describe it?',
        'images': ["data/images/unnamedjpg"]
    }],
    # options={"temperature":0.7}
    )

print(response)