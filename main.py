
# correct
import os
import time
import re
import json
import google.generativeai as genai
import pandas as pd


model = genai.GenerativeModel("gemini-2.0-flash")
AUTO_PROMPT = (
    "Прочти приведённый ниже текст и на его основе создай **информативный, глубокий и разноплановый датасет** "
    "из **8 пар 'вопрос-ответ'**, касающихся кыргызской истории, культуры, традиций и обычаев.\n\n"
    "**Важно:**\n"
    "- Все вопросы и ответы должны быть написаны **строго на кыргызском языке**.\n"
    "- Каждый **вопрос должен содержать не менее 2000 символов**.\n"
    "- Каждый **ответ должен содержать не менее 2000 символов**.\n\n"
    "**Формулировка вопросов:**\n"
    "- Не используй фразы вроде «текстке ылайык», «текстте берилгендей», «текстте», «тексттеги», «текстте айтылгандай», «жогорудагы маалыматтарга таянып», и т.д.\n"
    "- Каждый вопрос должен быть написан **непосредственно по теме**, **без отсылок на текст**.\n\n"
    "**Требования к вопросам:**\n"
    "- Вопросы должны быть чёткими, развернутыми и концептуальными, охватывать различные аспекты: тарых, инсандар, маданият, каада-салттар ж.б.\n"
    "- Не допускаются поверхностные или банальные вопросы.\n\n"
    "**Требования к ответам:**\n"
    "- Ответы должны быть логичными, фактологичными и хорошо структурированными.\n"
    "- По возможности включай мисалдар, даталар, контекст, салыштыруулар.\n\n"
    "**Формат:** Верни строго JSON-массив без пояснений. Пример:\n"
    "[\n  {\"question\": \"...\", \"answer\": \"...\"},\n  ...\n]"
)



def split_text_by_paragraphs(text: str, paragraphs_per_chunk: int = 60) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []

    for i in range(0, len(paragraphs), paragraphs_per_chunk):
        chunk = "\n".join(paragraphs[i:i + paragraphs_per_chunk])
        chunks.append(chunk)

    return chunks


def clean_response(text: str) -> str:
    return re.sub(r"```(?:json)?|```", "", text).strip()


def extract_qa_pairs(text: str) -> list[tuple[str, str]]:
    text = clean_response(text)

    try:
        data = json.loads(text)
        return [(item['question'], item['answer']) for item in data if 'question' in item and 'answer' in item]
    except json.JSONDecodeError:
        print("⚠️ JSON невалиден, пробуем извлечь вручную через регулярные выражения...")

    pattern = re.compile(
        r'{"question"\s*:\s*"(?P<question>.*?)"\s*,\s*"answer"\s*:\s*"(?P<answer>.*?)"}',
        re.DOTALL
    )

    matches = pattern.findall(text)
    if not matches:
        print("⚠️ Не удалось извлечь пары вручную.")
    return matches

def main():
    file_path = "data/Кыргыздын_кол_өнөрчүлүгү.txt"
    output_file = "Кыргыздын_кол_өнөрчүлүгү.xlsx"

    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = split_text_by_paragraphs(full_text, paragraphs_per_chunk=60)
    print(f"✅ Разбито на {len(chunks)} частей по 60 абзацев")


    data = []
    seen_questions = set()

    for idx, chunk in enumerate(chunks):
        print(f"\n🔄 Обработка части {idx + 1} из {len(chunks)}...")
        full_prompt = f"{AUTO_PROMPT}\n\nТекст:\n{chunk}"

        try:
            response = model.generate_content(full_prompt)
            cleaned = clean_response(response.text)
            qa_pairs = extract_qa_pairs(cleaned)

            if qa_pairs:
                for q, a in qa_pairs:
                    q = q.strip()
                    a = a.strip()
                    if q in seen_questions:
                        continue
                    seen_questions.add(q)
                    data.append({
                        "Вопрос": q,
                        "Ответ": a,
                        "Длина вопроса": len(q),
                        "Длина ответа": len(a)
                    })
                print(f"✅ Найдено пар: {len(qa_pairs)}")
            else:
                print("⚠️ Не удалось найти пары вопрос-ответ")

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            with open("errors.txt", "a", encoding="utf-8") as log:
                log.write(f"\n--- Часть {idx+1} ---\n{chunk}\nОшибка: {e}\n")

        time.sleep(30)

    if data:
        df = pd.DataFrame(data)
        df.to_excel(output_file, index=False)
        print(f"\n📁 Excel-файл сохранён: {output_file}")
    else:
        print("❌ Не удалось получить данные для сохранения.")



if __name__ == "__main__":
    main()

