from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import re
import nltk
from langdetect import detect, LangDetectException
import spacy

# Загрузка моделей (ленивая загрузка)
nlp_spacy = None

def get_spacy_model():
    """Ленивая загрузка spaCy модели"""
    global nlp_spacy
    if nlp_spacy is None:
        try:
            nlp_spacy = spacy.load("ru_core_news_sm")
        except OSError:
            print("Загрузка spaCy модели...")
            import subprocess
            import sys
            # Используем pip для установки модели напрямую
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.7.0/ru_core_news_sm-3.7.0-py3-none-any.whl"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    nlp_spacy = spacy.load("ru_core_news_sm")
                else:
                    raise Exception(f"Ошибка установки: {result.stderr}")
            except Exception as e:
                print(f"Ошибка загрузки spaCy модели: {e}")
                print("Используем базовую модель без предобученных компонентов")
                # Fallback: используем базовую модель
                nlp_spacy = spacy.blank("ru")
    return nlp_spacy

# Загрузка NLTK данных
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

app = FastAPI(title="NLP Service", version="1.0.0")


class ExtractClaimsRequest(BaseModel):
    text: str = Field(..., min_length=100)


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int


class Claim(BaseModel):
    text: str
    importance: float = 1.0
    entities: List[str] = []


class ExtractClaimsResponse(BaseModel):
    claims: List[Claim]
    entities: List[Entity]
    language: str
    normalized_text: str


def preprocess_text(text: str) -> str:
    """Предобработка текста"""
    # Удаление HTML тегов
    text = re.sub(r'<[^>]+>', '', text)
    # Нормализация пробелов
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # Нормализация кавычек
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    return text


def detect_language(text: str) -> str:
    """Определение языка текста"""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "ru"  # По умолчанию русский


def extract_entities(text: str) -> List[Entity]:
    """Извлечение именованных сущностей с помощью spaCy"""
    nlp = get_spacy_model()
    doc = nlp(text)
    entities = []
    
    # Если модель не загрузилась или не имеет NER, используем простую эвристику
    if not hasattr(doc, 'ents') or len(doc.ents) == 0:
        # Простая эвристика для извлечения сущностей
        import re
        words = text.split()
        for i, word in enumerate(words):
            # Имена собственные (с заглавной буквы, не в начале предложения)
            if word and word[0].isupper() and len(word) > 2:
                # Проверяем что это не начало предложения
                if i > 0 and words[i-1][-1] not in '.!?':
                    start_pos = text.find(word)
                    if start_pos >= 0:
                        entities.append(Entity(
                            text=word,
                            type="PER",
                            start=start_pos,
                            end=start_pos + len(word)
                        ))
        
        # Поиск дат
        date_pattern = r'\d{1,2}[./]\d{1,2}[./]\d{2,4}'
        for match in re.finditer(date_pattern, text):
            entities.append(Entity(
                text=match.group(),
                type="DATE",
                start=match.start(),
                end=match.end()
            ))
    else:
        # Используем результаты spaCy
        for ent in doc.ents:
            # Маппинг типов spaCy на наши типы
            entity_type = ent.label_
            if entity_type == "PERSON":
                entity_type = "PER"
            elif entity_type == "ORG":
                entity_type = "ORG"
            elif entity_type == "GPE" or entity_type == "LOC":
                entity_type = "LOC"
            elif entity_type == "DATE":
                entity_type = "DATE"
            elif entity_type == "MONEY":
                entity_type = "MONEY"
            else:
                entity_type = "MISC"
            
            entities.append(Entity(
                text=ent.text,
                type=entity_type,
                start=ent.start_char,
                end=ent.end_char
            ))
    
    return entities


def is_factual_sentence(sentence: str, entities: List[Entity]) -> bool:
    """Определение, является ли предложение фактическим утверждением"""
    sentence_lower = sentence.lower()
    
    # Признаки фактического утверждения
    factual_indicators = [
        r'\d+',  # Содержит числа
        r'\d{1,2}[./]\d{1,2}[./]\d{2,4}',  # Даты
        r'произошло|состоялось|объявил|сообщил|заявил|подтвердил',
        r'миллион|миллиард|тысяч|процент',
    ]
    
    # Проверка наличия индикаторов
    has_indicators = any(re.search(pattern, sentence_lower) for pattern in factual_indicators)
    
    # Проверка наличия сущностей в предложении
    has_entities = any(
        entity.start >= sentence.find(sentence) and entity.end <= sentence.find(sentence) + len(sentence)
        for entity in entities
    )
    
    # Исключаем вопросы и мнения
    is_question = sentence.strip().endswith('?')
    opinion_indicators = ['думаю', 'считаю', 'мне кажется', 'возможно', 'вероятно']
    is_opinion = any(indicator in sentence_lower for indicator in opinion_indicators)
    
    return (has_indicators or has_entities) and not is_question and not is_opinion


def extract_claims(text: str, entities: List[Entity]) -> List[Claim]:
    """Извлечение утверждений (claims) из текста"""
    # Разбивка на предложения
    sentences = nltk.sent_tokenize(text, language='russian')
    
    claims = []
    entity_texts = {e.text.lower() for e in entities}
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:  # Пропускаем слишком короткие предложения
            continue
        
        if is_factual_sentence(sentence, entities):
            # Извлечение упомянутых сущностей в предложении
            mentioned_entities = [
                e.text for e in entities
                if e.text.lower() in sentence.lower()
            ]
            
            # Оценка важности на основе длины и наличия сущностей
            importance = 0.5
            if mentioned_entities:
                importance += 0.3
            if re.search(r'\d+', sentence):
                importance += 0.2
            
            claims.append(Claim(
                text=sentence,
                importance=min(importance, 1.0),
                entities=mentioned_entities
            ))
    
    # Сортировка по важности
    claims.sort(key=lambda x: x.importance, reverse=True)
    
    # Ограничение количества claims (топ-10)
    return claims[:10]


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "nlp_service"}


@app.post("/extract-claims", response_model=ExtractClaimsResponse)
async def extract_claims_endpoint(request: ExtractClaimsRequest):
    """
    Извлечение утверждений и сущностей из текста.
    
    Процесс:
    1. Предобработка текста
    2. Определение языка
    3. Извлечение именованных сущностей (NER)
    4. Извлечение фактических утверждений (claims)
    """
    try:
        # Предобработка
        normalized_text = preprocess_text(request.text)
        
        # Определение языка
        language = detect_language(normalized_text)
        
        if language != "ru":
            # Для не-русского текста возвращаем предупреждение
            # В продакшене можно добавить поддержку других языков
            pass
        
        # Извлечение сущностей
        entities = extract_entities(normalized_text)
        
        # Извлечение утверждений
        claims = extract_claims(normalized_text, entities)
        
        return ExtractClaimsResponse(
            claims=claims,
            entities=entities,
            language=language,
            normalized_text=normalized_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

