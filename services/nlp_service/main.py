from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import os, re, subprocess, sys, nltk, spacy
from langdetect import detect, LangDetectException

app = FastAPI(title="NLP Service", version="1.0.0")
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "12"))
MIN_CLAIM_CHARS = int(os.getenv("MIN_CLAIM_CHARS", "25"))
MAX_CLAIM_CHARS = int(os.getenv("MAX_CLAIM_CHARS", "280"))
MIN_CLAIM_TOKENS = int(os.getenv("MIN_CLAIM_TOKENS", "6"))
MIN_FACTUAL_SCORE = float(os.getenv("MIN_FACTUAL_SCORE", "0.45"))
REQUIRE_FACTUAL_VERB = os.getenv("REQUIRE_FACTUAL_VERB", "true").lower() in {"1", "true", "yes"}
STRICT_ATTRIBUTION_FILTER = os.getenv("STRICT_ATTRIBUTION_FILTER", "true").lower() in {"1", "true", "yes"}
FACTUAL_VERBS = r"произошл|состоял|объяв|сообщ|заяв|подтверд|отметил|подчеркнул|подписал|ввели|принял|одобрил|опубликовал|представил|увеличил|снизил|достиг|зарегистрировал|зафиксировал"
OPINION_PATTERNS = r"думаю|считаю|мне кажется|вероятно|возможно|по моему мнению|скорее всего|может быть"
CLAUSE_SPLIT_PATTERN = r";|\s(?:однако|при этом|в то же время|между тем|позже|затем|после этого)\s|,\s(?:а также|однако|при этом|но)\s"
REPORTING_VERBS = {"сказать", "заявить", "сообщить", "отметить", "подчеркнуть", "добавить"}
FACTUAL_VERB_FALLBACK_PATTERN = re.compile(
    r"\b("
    r"предложил(?:а|и)?|"
    r"заявил(?:а|и)?|"
    r"сообщил(?:а|и)?|"
    r"объявил(?:а|и)?|"
    r"подтвердил(?:а|и)?|"
    r"опубликовал(?:а|и)?|"
    r"признал(?:а|и)?|"
    r"подписал(?:а|и)?|"
    r"принял(?:а|и)?|"
    r"включил(?:а|и)?|"
    r"ограничил(?:а|и)?|"
    r"запретил(?:а|и)?|"
    r"разрешил(?:а|и)?|"
    r"начал(?:а|и)?|"
    r"завершил(?:а|и)?"
    r")\b",
    flags=re.IGNORECASE,
)
nlp_spacy = None


def get_spacy_model():
    global nlp_spacy
    if nlp_spacy is None:
        try:
            nlp_spacy = spacy.load("ru_core_news_sm")
        except OSError:
            print("Загрузка spaCy модели...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.7.0/ru_core_news_sm-3.7.0-py3-none-any.whl"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                nlp_spacy = spacy.load("ru_core_news_sm") if result.returncode == 0 else spacy.blank("ru")
            except Exception:
                nlp_spacy = spacy.blank("ru")
        if "parser" not in nlp_spacy.pipe_names and "senter" not in nlp_spacy.pipe_names and "sentencizer" not in nlp_spacy.pipe_names:
            nlp_spacy.add_pipe("sentencizer")
    return nlp_spacy


NLTK_PUNKT_AVAILABLE = True
try:
    nltk.data.find("tokenizers/punkt/PY3/russian.pickle")
except LookupError:
    NLTK_PUNKT_AVAILABLE = nltk.download("punkt", quiet=True)


def regex_split_sentences(text: str) -> List[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


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
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("«", "\"").replace("»", "\"").replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", text.lower().replace("ё", "е"))).strip()


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "ru"


def split_sentences(text: str) -> List[str]:
    nlp = get_spacy_model()
    try:
        doc = nlp(text)
        sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
        if sentences:
            return sentences
    except Exception:
        pass
    if NLTK_PUNKT_AVAILABLE:
        try:
            return [sentence.strip() for sentence in nltk.sent_tokenize(text, language="russian") if sentence.strip()]
        except Exception:
            pass
    return regex_split_sentences(text)


def split_atomic_clauses(sentence: str) -> List[str]:
    clauses = [sentence.strip(" ,;-") for sentence in re.split(CLAUSE_SPLIT_PATTERN, sentence) if sentence.strip(" ,;-")]
    refined = []
    for clause in clauses:
        if len(clause) > MAX_CLAIM_CHARS and ", " in clause:
            refined.extend([part.strip(" ,") for part in clause.split(", ") if part.strip(" ,")])
        else:
            refined.append(clause)
    return refined


def extract_entities(text: str) -> List[Entity]:
    nlp = get_spacy_model()
    doc = nlp(text)
    entities, seen = [], set()
    if not hasattr(doc, "ents") or len(doc.ents) == 0:
        for match in re.finditer(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{4}\s*года?", text):
            key = (match.group(), "DATE", match.start(), match.end())
            if key not in seen:
                entities.append(Entity(text=match.group(), type="DATE", start=match.start(), end=match.end()))
                seen.add(key)
        for match in re.finditer(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2}\b", text):
            key = (match.group(), "MISC", match.start(), match.end())
            if key not in seen:
                entities.append(Entity(text=match.group(), type="MISC", start=match.start(), end=match.end()))
                seen.add(key)
        return entities
    for ent in doc.ents:
        entity_type = {"PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC", "DATE": "DATE", "MONEY": "MONEY"}.get(ent.label_, "MISC")
        key = (ent.text, entity_type, ent.start_char, ent.end_char)
        if key not in seen:
            entities.append(Entity(text=ent.text, type=entity_type, start=ent.start_char, end=ent.end_char))
            seen.add(key)
    return entities


def get_sentence_entities(sentence: str, entities: List[Entity]) -> List[str]:
    normalized_sentence = normalize_text(sentence)
    matched = []
    for entity in entities:
        if normalize_text(entity.text) and normalize_text(entity.text) in normalized_sentence:
            matched.append(entity.text)
    return list(dict.fromkeys(matched))


def has_factual_verb_signal(text: str, has_finite_verb: bool = False) -> bool:
    return bool(re.search(FACTUAL_VERBS, text)) or bool(FACTUAL_VERB_FALLBACK_PATTERN.search(text)) or has_finite_verb


def factual_score(sentence: str, matched_entities: List[str]) -> float:
    lower = sentence.lower()
    score = 0.0
    if re.search(r"\d", lower):
        score += 0.25
    if re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{4}\s*года?", lower):
        score += 0.15
    if has_factual_verb_signal(lower):
        score += 0.20
    if re.search(r"миллион|миллиард|тысяч|процент|рубл|доллар|евро", lower):
        score += 0.15
    if matched_entities:
        score += min(0.25, 0.08 * len(matched_entities))
    if sentence[0].isupper():
        score += 0.05
    return min(score, 1.0)


def claim_quality_signals(sentence: str) -> Dict[str, bool]:
    signals = {
        "has_finite_verb": False,
        "has_subject": False,
        "has_object_or_complement": False,
        "starts_with_reporting_verb": False,
        "has_reporting_complement": False,
    }
    try:
        doc = get_spacy_model()(sentence)
        tokens = [t for t in doc if not t.is_space]
        if not tokens:
            return signals

        for t in tokens:
            if t.pos_ in {"VERB", "AUX"} and ("Fin" in str(t.morph.get("VerbForm")) or t.dep_ == "ROOT"):
                signals["has_finite_verb"] = True
            if t.dep_ in {"nsubj", "csubj", "nsubj:pass"}:
                signals["has_subject"] = True
            if t.dep_ in {"obj", "iobj", "obl", "xcomp", "ccomp"}:
                signals["has_object_or_complement"] = True

        first_lemma = tokens[0].lemma_.lower() if tokens[0].lemma_ else tokens[0].text.lower()
        if first_lemma in REPORTING_VERBS:
            signals["starts_with_reporting_verb"] = True

        if any(t.lower_ == "что" for t in tokens) or any(t.dep_ in {"xcomp", "ccomp"} for t in tokens):
            signals["has_reporting_complement"] = True

        return signals
    except Exception:
        return signals


def is_factual_claim(sentence: str, matched_entities: List[str]) -> bool:
    lower = sentence.lower()
    if sentence.endswith("?") or len(sentence) < MIN_CLAIM_CHARS or len(sentence) > MAX_CLAIM_CHARS:
        return False
    if len(re.findall(r"\w+", sentence, flags=re.UNICODE)) < MIN_CLAIM_TOKENS:
        return False
    if re.search(OPINION_PATTERNS, lower):
        return False
    if re.search(r"\bесли\b|\bмог\b|\bмогут\b|\bможет\b", lower):
        return False
    if lower.startswith(("по данным", "как сообщалось ранее")) and len(matched_entities) == 0:
        return False
    signals = claim_quality_signals(sentence)
    factual_verb_signal = has_factual_verb_signal(lower, signals["has_finite_verb"])
    if REQUIRE_FACTUAL_VERB and not factual_verb_signal:
        return False
    if (
        STRICT_ATTRIBUTION_FILTER
        and signals["starts_with_reporting_verb"]
        and not signals["has_reporting_complement"]
        and len(matched_entities) == 0
    ):
        return False
    if not (signals["has_subject"] or matched_entities or re.search(r"\d", lower)):
        return False
    if not (signals["has_object_or_complement"] or matched_entities or re.search(r"\d", lower)):
        return False
    score = factual_score(sentence, matched_entities)
    if factual_verb_signal and len(matched_entities) >= 2:
        score += 0.10
    return score >= MIN_FACTUAL_SCORE


def clean_claim_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -,:;.")
    return text[0].upper() + text[1:] if text else text


def extract_claims(text: str, entities: List[Entity]) -> List[Claim]:
    claims, seen = [], set()
    for sentence in split_sentences(text):
        for clause in split_atomic_clauses(sentence):
            clause = clean_claim_text(clause)
            if len(clause) < MIN_CLAIM_CHARS:
                continue
            matched_entities = get_sentence_entities(clause, entities)
            if not is_factual_claim(clause, matched_entities):
                continue
            normalized_clause = normalize_text(clause)
            if normalized_clause in seen:
                continue
            seen.add(normalized_clause)
            importance = max(0.35, factual_score(clause, matched_entities))
            claims.append(Claim(text=clause, importance=importance, entities=matched_entities))
    claims.sort(key=lambda claim: (claim.importance, len(claim.entities), -len(claim.text)), reverse=True)
    return claims[:MAX_CLAIMS]


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "nlp_service",
        "max_claims": MAX_CLAIMS,
        "min_claim_chars": MIN_CLAIM_CHARS,
        "min_claim_tokens": MIN_CLAIM_TOKENS,
        "min_factual_score": MIN_FACTUAL_SCORE,
        "require_factual_verb": REQUIRE_FACTUAL_VERB,
        "strict_attribution_filter": STRICT_ATTRIBUTION_FILTER,
    }


@app.post("/extract-claims", response_model=ExtractClaimsResponse)
async def extract_claims_endpoint(request: ExtractClaimsRequest):
    try:
        normalized_text = preprocess_text(request.text)
        language = detect_language(normalized_text)
        entities = extract_entities(normalized_text)
        claims = extract_claims(normalized_text, entities)
        return ExtractClaimsResponse(claims=claims, entities=entities, language=language, normalized_text=normalized_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
