from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME
from embed import search


client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

messages=[
    {
        "role": "system",
        "content": "존대말 사용. 정보 탐색 질문이 아닐경우 [문서]를 참고하지 말고 [질문]에만 답해라."
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "현재 시간을 알려준다",
            "parameters": {}
        }
    }
]


def get_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


while True:
    user_input = input("나 > ")
    if user_input.lower() in ["exit", "quit"]:
        break

    
    # 🔥 1. 벡터 검색
    docs = search(user_input)

    context = "\n\n".join(docs)

    # 🔥 2. RAG 프롬프트
    rag_prompt = f"""
아래 문서를 참고해서 질문에 답해라.
문서에 없는 내용은 추측하지 마라.

[문서]
{context}

[질문]
{user_input}
"""

    messages.append({"role": "user", "content": rag_prompt})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        # tools=tools,
        # tool_choice="auto"
    )

    ai_message = response.choices[0].message
    messages.append({"role": "assistant", "content": ai_message.content})



    if ai_message.tool_calls:
        tool_call = ai_message.tool_calls[0]
        tool_name = tool_call.function.name

        if tool_name == "get_time":
            result = get_time()

            # 도구 결과를 messages에 다시 추가
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

            # AI를 한 번 더 호출
            second_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )

            final_message = second_response.choices[0].message.content
            print("AI >", final_message)

    else:
        print("AI >", ai_message.content)

