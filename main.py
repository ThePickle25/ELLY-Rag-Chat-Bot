# from langchain_openai import ChatOpenAI
# from openai import OpenAI
import json
#
# GEMINI_KEY = 'AIzaSyBjYWfBi3Yo2dT46Maw6sc0hzIxyoaicn8'
#
# client = OpenAI(
#     api_key=GEMINI_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )
#
# response = client.chat.completions.create(
#     model="gemini-2.0-flash",
#     messages=[
#         {"role": "system", "content": "You are a book expert who describes fictional characters in a short paragraph."},
#
#         # Few-shot Example 1
#         {"role": "user", "content": "Tell me about Guo Jing."},
#         {"role": "assistant",
#          "content": "Guo Jing is the main character in 'The Legend of the Condor Heroes' by Jin Yong. He is known for "
#                     "his honesty, bravery, and martial arts skills."},
#
#         # Few-shot Example 2
#         {"role": "user", "content": "Tell me about Frodo Baggins."},
#         {"role": "assistant",
#          "content": "Frodo Baggins is a hobbit from 'The Lord of the Rings' by J.R.R. Tolkien. He is chosen to carry "
#                     "the One Ring to Mount Doom and destroy it, showing great courage and resilience."},
#
#         # Actual question
#         {"role": "user", "content": "Tell me about Ron Weasley."}
#     ]
# )
#
# print(response.choices[0].message)
with open("doc.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("output.txt", "w", encoding="utf-8") as f:
    for item in data:
        line = f"{item['page_content']}: {item['metadata']}\n"
        f.write(line)
