from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

app = FastAPI(title="Verifier Service", version="1.0.0")

# Конфигурация
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# Ленивая загрузка моделей
embedding_model = None
nli_model = None
USE_CROSS_ENCODER = False

def get_embedding_model():
    """Ленивая загрузка модели эмбеддингов"""
    global embedding_model
    if embedding_model is None:
        print("Загрузка модели эмбеддингов для верификации...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return embedding_model

def get_nli_model():
    """Ленивая загрузка NLI модели"""
    global nli_model, USE_CROSS_ENCODER
    if nli_model is None and not USE_CROSS_ENCODER:
        # Пытаемся загрузить cross-encoder
        try:
            from sentence_transformers import CrossEncoder
            print("Загрузка Cross-encoder модели...")
            nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
            USE_CROSS_ENCODER = True
        except Exception:
            print("Cross-encoder недоступен, используем семантическое сходство")
            USE_CROSS_ENCODER = False
    return nli_model


class Claim(BaseModel):
    text: str
    importance: float = 1.0
    entities: List[str] = []


class Article(BaseModel):
    url: str
    domain: str
    title: str
    date: Optional[str] = None
    text: str
    chunks: List[str] = []
    trust_level: float = 1.0


class VerifyRequest(BaseModel):
    claims: List[Claim]
    articles: List[Article]
    original_text: str


class Evidence(BaseModel):
    url: str
    title: str
    date: Optional[str] = None
    snippet: str


class ClaimResult(BaseModel):
    claim: str
    label: str  # ENTAILS, CONTRADICTS, NEUTRAL
    confidence: float
    evidence: List[Evidence] = []


class SourceInfo(BaseModel):
    url: str
    domain: str
    date: Optional[str] = None
    trust_level: float


class VerifyResponse(BaseModel):
    status: str
    confidence: float
    claim_results: List[ClaimResult]
    sources: List[SourceInfo]
    warnings: List[str] = []


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Вычисление семантического сходства между двумя текстами"""
    model = get_embedding_model()
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return float(similarity)


def classify_entailment(claim: str, evidence_text: str) -> tuple[str, float]:
    """
    Классификация отношения между claim и evidence.
    Возвращает (label, confidence)
    """
    model = get_nli_model()
    if model and USE_CROSS_ENCODER:
        # Используем cross-encoder для более точной классификации
        # Формат: [premise, hypothesis]
        try:
            scores = model.predict([[evidence_text, claim]])
            # scores: [entailment, neutral, contradiction] или просто числа
            # Для упрощения используем семантическое сходство если формат неожиданный
            if isinstance(scores, (list, np.ndarray)) and len(scores) >= 3:
                # Если три значения - берем максимальное
                max_idx = np.argmax(scores)
                if max_idx == 0:  # entailment
                    return "ENTAILS", float(scores[0])
                elif max_idx == 2:  # contradiction
                    return "CONTRADICTS", float(scores[2])
                else:
                    return "NEUTRAL", float(scores[1])
        except Exception as e:
            print(f"Ошибка при использовании cross-encoder: {e}, используем семантическое сходство")
    
    # Используем семантическое сходство как приближение
    similarity = compute_semantic_similarity(claim, evidence_text)
    
    # Эвристика для определения label
    # В продакшене нужна обученная NLI модель
    if similarity > 0.7:
        return "ENTAILS", similarity
    elif similarity < 0.3:
        # Проверка на противоречие (упрощенная)
        contradiction_keywords = [
            "не", "нет", "отрицает", "опровергает", "неверно", "ложь"
        ]
        has_contradiction = any(
            keyword in evidence_text.lower() or keyword in claim.lower()
            for keyword in contradiction_keywords
        )
        if has_contradiction:
            return "CONTRADICTS", 1.0 - similarity
        else:
            return "NEUTRAL", 0.5
    else:
        return "NEUTRAL", 0.5


def find_best_evidence(claim: str, article: Article, top_k: int = 3) -> List[Evidence]:
    """Поиск лучших фрагментов статьи для подтверждения claim"""
    if not article.chunks:
        # Если чанки не предоставлены, разбиваем текст
        words = article.text.split()
        chunk_size = 100
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    else:
        chunks = article.chunks
    
    # Вычисляем сходство для каждого чанка
    chunk_scores = []
    for chunk in chunks:
        similarity = compute_semantic_similarity(claim, chunk)
        chunk_scores.append((chunk, similarity))
    
    # Сортируем по сходству
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Берем топ-K
    evidence_list = []
    for chunk, score in chunk_scores[:top_k]:
        if score > 0.5:  # Минимальный порог релевантности
            evidence_list.append(
                Evidence(
                    url=article.url,
                    title=article.title,
                    date=article.date,
                    snippet=chunk[:200] + "..." if len(chunk) > 200 else chunk
                )
            )
    
    return evidence_list


def verify_claims(claims: List[Claim], articles: List[Article]) -> Dict[str, Any]:
    """Верификация всех claims против статей"""
    claim_results = []
    all_sources = {}
    
    for claim in claims:
        best_entails = None
        best_contradicts = None
        best_entails_score = 0.0
        best_contradicts_score = 0.0
        best_entails_evidence = []
        best_contradicts_evidence = []
        
        for article in articles:
            # Поиск лучших фрагментов
            evidence_list = find_best_evidence(claim.text, article, top_k=2)
            
            if not evidence_list:
                continue
            
            # Классификация для лучшего фрагмента
            best_evidence = evidence_list[0]
            label, confidence = classify_entailment(claim.text, best_evidence.snippet)
            
            if label == "ENTAILS" and confidence > best_entails_score:
                best_entails = article
                best_entails_score = confidence
                best_entails_evidence = evidence_list[:2]  # Топ-2 фрагмента
            
            elif label == "CONTRADICTS" and confidence > best_contradicts_score:
                best_contradicts = article
                best_contradicts_score = confidence
                best_contradicts_evidence = evidence_list[:2]
            
            # Сохраняем источник
            all_sources[article.url] = {
                "url": article.url,
                "domain": article.domain,
                "date": article.date,
                "trust_level": article.trust_level
            }
        
        # Определение финального label для claim
        if best_contradicts_score > 0.6:
            final_label = "CONTRADICTS"
            final_confidence = best_contradicts_score
            final_evidence = best_contradicts_evidence
        elif best_entails_score > 0.6:
            final_label = "ENTAILS"
            final_confidence = best_entails_score
            final_evidence = best_entails_evidence
        else:
            final_label = "NEUTRAL"
            final_confidence = max(best_entails_score, best_contradicts_score) if (best_entails_score > 0 or best_contradicts_score > 0) else 0.3
            final_evidence = best_entails_evidence if best_entails_evidence else best_contradicts_evidence
        
        claim_results.append(
            ClaimResult(
                claim=claim.text,
                label=final_label,
                confidence=final_confidence,
                evidence=final_evidence if final_evidence else []
            )
        )
    
    return {
        "claim_results": [cr.dict() for cr in claim_results],
        "sources": list(all_sources.values())
    }


def compute_overall_confidence(claim_results: List[ClaimResult]) -> float:
    """Вычисление общей уверенности на основе результатов по claims"""
    if not claim_results:
        return 0.0
    
    # Взвешенное среднее по важности и confidence
    total_weight = 0.0
    weighted_sum = 0.0
    
    for cr in claim_results:
        # Вес зависит от label
        if cr.label == "ENTAILS":
            weight = 1.0
        elif cr.label == "CONTRADICTS":
            weight = 0.5  # Противоречия снижают уверенность
        else:
            weight = 0.3
        
        weighted_sum += cr.confidence * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return min(weighted_sum / total_weight, 1.0)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "verifier_service"}


@app.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest):
    """
    Верификация claims против найденных статей.
    
    Процесс:
    1. Для каждого claim находим релевантные фрагменты в статьях
    2. Классифицируем отношение (ENTAILS/CONTRADICTS/NEUTRAL)
    3. Агрегируем результаты
    """
    try:
        if not request.claims:
            return VerifyResponse(
                status="INSUFFICIENT_DATA",
                confidence=0.0,
                claim_results=[],
                sources=[],
                warnings=["Нет утверждений для проверки"]
            )
        
        if not request.articles:
            return VerifyResponse(
                status="INSUFFICIENT_DATA",
                confidence=0.0,
                claim_results=[],
                sources=[],
                warnings=["Не найдено статей для проверки"]
            )
        
        # Верификация
        results = verify_claims(request.claims, request.articles)
        
        claim_results = [ClaimResult(**cr) for cr in results["claim_results"]]
        sources = [SourceInfo(**s) for s in results["sources"]]
        
        # Вычисление общей уверенности
        overall_confidence = compute_overall_confidence(claim_results)
        
        # Определение статуса
        entails_count = sum(1 for cr in claim_results if cr.label == "ENTAILS")
        contradicts_count = sum(1 for cr in claim_results if cr.label == "CONTRADICTS")
        total = len(claim_results)
        
        if contradicts_count > total * 0.3:
            status = "REFUTED"
        elif entails_count >= total * 0.6:
            status = "CONFIRMED"
        elif entails_count > 0:
            status = "PARTIALLY_CONFIRMED"
        else:
            status = "INSUFFICIENT_DATA"
        
        warnings = []
        if status == "INSUFFICIENT_DATA":
            warnings.append("Недостаточно данных для проверки утверждений")
        elif status == "PARTIALLY_CONFIRMED":
            warnings.append("Часть утверждений не подтверждена")
        
        return VerifyResponse(
            status=status,
            confidence=overall_confidence,
            claim_results=claim_results,
            sources=sources,
            warnings=warnings
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка верификации: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

