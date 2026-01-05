from openai import OpenAI
import os

def create_client():
    # Read API key from environment variable (recommended practice)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable OPENAI_API_KEY not found. "
            "Please set it in your system or current shell session."
        )
    return OpenAI(api_key=api_key)

def chat_loop():
    client = create_client()

    print("=== Simple AI Agent Chatbox ===")
    print("exit or quit\n")

    # System prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Behave like an AI agent that helps the user step-by-step, "
                "asking brief clarifying questions only when absolutely necessary."
            ),
        }
    ]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("AI: Bye~")
            break

        # Record user message
        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  
                messages=messages,
            )
        except Exception as e:
            print(f"[Error] OpenAI API call failed: {e}")
            continue

        # Extract model reply
        msg_obj = response.choices[0].message
        # In the new SDK, message is an object; use .content to read the text
        assistant_reply = msg_obj.content

        print(f"AI: {assistant_reply}\n")

        # Record assistant reply to maintain conversation history
        messages.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    chat_loop()
