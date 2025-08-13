# CORRECT
import os
import time
import re
import json
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("API ключ не найден. Проверьте файл .env")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

AUTO_PROMPT = (
    "Берилген тексттин негизинде кыргыз элинин тарыхына, тилине, географиясына, экономикасына, маданиятына, салт-санаасына, дүйнө таанымына жана башка ушул сыяктуу аспектилерине байланыштуу терең, маалыматтуу жана сапаттуу датасетти түз, ал 7 жуп «суроо-жооптон» турсун.\n\n"
    "Тили:\n"
    "- Бардык суроолор жана жооптор кыргыз тилинде гана жазылышы керек.\n\n"
    "Суроолорго коюлган талаптар:\n"
    "- Ар бир суроо кеминде 1000 символдон турушу керек.\n"
    "- Суроо логикалык жактан толук, кеңири жайылган, теманын 1-2 гана өз ара байланышкан аспектисин камтышы керек.\n"
    "- Суроолор так, тематикалык жактан фокусталган, ар кандай темалар менен ашыкча жүктөлбөгөн болушу керек.\n"
    "- Керек болсо, тактоочу деталдарды же байланышкан пункттарды кошууга болот, бирок алардын бардыгы бир негизги суроого тиешелүү болушу керек.\n"
    "- «Текстке ылайык», «текстте берилгендей», «текстте», «тексттеги», «текстте айтылгандай», «жогорудагы маалыматтарга таянып» сыяктуу сөз айкаштарын жана ушуга окшош шилтемелерди колдонбо.\n\n"
    "Жоопторго коюлган талаптар:\n"
    "- Ар бир жооп кеминде 1000 символдон турушу керек.\n"
    "- Жооптор логикалык, фактыларга негизделген, мисалдар, тарыхый жана этнографиялык деталдар менен берилиши керек.\n"
    "- Жооп берилген суроону толук жана терең ачып бериши керек.\n\n"
    "- «Текстке ылайык», «текстте берилгендей», «текстте», «тексттеги», «текстте айтылгандай», «жогорудагы маалыматтарга таянып» сыяктуу сөз айкаштарын жана ушуга окшош шилтемелерди колдонбо.\n\n"
    "Формат: Эч кандай кошумча түшүндүрмөсү жок JSON-массивди гана кайтар. Мисалы:\n"
    "[\n  {«question«: «...», «answer»:«...»},\n  ...\n]"
)


def split_text_by_paragraphs(text: str, paragraphs_per_chunk: int = 50) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return ["\n".join(paragraphs[i:i + paragraphs_per_chunk])
            for i in range(0, len(paragraphs), paragraphs_per_chunk)]


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
    return matches

def main():
    file_path = "data/Адабият теориясы(okuma.kg)_Кыргыз тили жана адабияты71.txt"
    output_file = "Адабият теориясы(okuma.kg)_Кыргыз тили жана адабияты71.xlsx"

    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    print(f"📏 Общая длина текста: {len(full_text):,} символов")

    chunks = split_text_by_paragraphs(full_text, paragraphs_per_chunk=50)
    print(f"✅ Разбито на {len(chunks)} частей по 50 абзацев")

    # Загрузка уже сохранённых данных, если файл существует
    if os.path.exists(output_file):
        df = pd.read_excel(output_file)
        seen_questions = set(df["Вопрос"].tolist())
        processed_count = len(df)
        data = df.to_dict("records")
        print(f"🔁 Продолжаем с части {processed_count + 1}")
    else:
        seen_questions = set()
        processed_count = 0
        data = []

    for idx, chunk in enumerate(chunks):
        if idx < processed_count:
            print(f"⏭️ Пропуск части {idx + 1} (уже обработана)")
            continue

        print(f"\n🔄 Обработка части {idx + 1} из {len(chunks)}...")
        full_prompt = f"{AUTO_PROMPT}\n\nТекст:\n{chunk}"

        try:
            response = model.generate_content(full_prompt)
            cleaned = clean_response(response.text)
            qa_pairs = extract_qa_pairs(cleaned)

            if qa_pairs:
                new_rows = []
                for q, a in qa_pairs:
                    q = q.strip()
                    a = a.strip()
                    if q in seen_questions:
                        continue
                    seen_questions.add(q)
                    new_rows.append({
                        "Вопрос": q,
                        "Ответ": a,
                        "Длина вопроса": len(q),
                        "Длина ответа": len(a)
                    })
                data.extend(new_rows)

                # 💾 Автосохранение после каждой части
                pd.DataFrame(data).to_excel(output_file, index=False)
                print(f"✅ Добавлено {len(new_rows)} новых пар, файл обновлён.")
            else:
                print("⚠️ Не удалось найти пары вопрос-ответ")

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            with open("errors.txt", "a", encoding="utf-8") as log:
                log.write(f"\n--- Часть {idx + 1} ---\n{chunk}\nОшибка: {e}\n")

        time.sleep(30)

    print(f"\n📁 Готово. Файл сохранён: {output_file}")


if __name__ == "__main__":
    main()