from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import os
from datetime import datetime
from enum import Enum

from database import init_db

app = FastAPI(
    title="Fake Detector API",
    description="API для проверки достоверности информации в новостных текстах",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация БД при старте
@app.on_event("startup")
async def startup_event():
    await init_db()

# Конфигурация сервисов
NLP_SERVICE_URL = os.getenv("NLP_SERVICE_URL", "http://nlp_service:8001")
SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "http://search_service:8002")
VERIFIER_SERVICE_URL = os.getenv("VERIFIER_SERVICE_URL", "http://verifier_service:8003")


class VerificationStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    PARTIALLY_CONFIRMED = "PARTIALLY_CONFIRMED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ClaimEvidence(BaseModel):
    url: str
    title: str
    date: str
    snippet: str


class ClaimResult(BaseModel):
    claim: str
    label: str  # ENTAILS, CONTRADICTS, NEUTRAL
    evidence: List[ClaimEvidence] = []


class SourceInfo(BaseModel):
    url: str
    domain: str
    date: str
    trust_level: float


class VerifyRequest(BaseModel):
    text: str = Field(..., min_length=300, description="Текст новости для проверки (минимум 300 символов)")


class VerifyResponse(BaseModel):
    status: VerificationStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    claims: List[ClaimResult] = []
    sources: List[SourceInfo] = []
    warnings: List[str] = []
    request_id: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Fake Detector API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/verify", response_model=VerifyResponse)
async def verify_text(
    request: VerifyRequest,
    background_tasks: BackgroundTasks
):
    """
    Проверка достоверности текста новости.
    
    Процесс:
    1. Предобработка и извлечение утверждений (NLP Service)
    2. Поиск релевантных статей (Search Service)
    3. Верификация фактов (Verifier Service)
    4. Формирование результата
    """
    try:
        # Создаем клиент с увеличенным таймаутом для всех запросов
        # Увеличенный таймаут для первого запроса (загрузка моделей может занять 5+ минут)
        timeout = httpx.Timeout(600.0, connect=30.0)  # 10 минут на запрос, 30 секунд на подключение
        
        # Шаг 1: Предобработка и извлечение утверждений
        async with httpx.AsyncClient(timeout=timeout) as client:
            nlp_response = await client.post(
                f"{NLP_SERVICE_URL}/extract-claims",
                json={"text": request.text}
            )
            nlp_response.raise_for_status()
            nlp_data = nlp_response.json()
        
        claims = nlp_data.get("claims", [])
        entities = nlp_data.get("entities", [])
        
        if not claims:
            return VerifyResponse(
                status=VerificationStatus.INSUFFICIENT_DATA,
                confidence=0.0,
                warnings=["Не удалось извлечь проверяемые утверждения из текста"]
            )
        
        # Преобразуем в словари для сериализации
        claims_dict = [c if isinstance(c, dict) else c.dict() for c in claims]
        entities_dict = [e if isinstance(e, dict) else e.dict() for e in entities]
        
        # Шаг 2: Поиск релевантных статей
        async with httpx.AsyncClient(timeout=timeout) as client:
            search_response = await client.post(
                f"{SEARCH_SERVICE_URL}/search",
                json={
                    "claims": claims_dict,
                    "entities": entities_dict,
                    "text": request.text
                }
            )
            search_response.raise_for_status()
            search_data = search_response.json()
        
        found_articles = search_data.get("articles", [])
        
        if not found_articles:
            return VerifyResponse(
                status=VerificationStatus.INSUFFICIENT_DATA,
                confidence=0.0,
                warnings=["Не найдено релевантных статей в доверенных источниках"]
            )
        
        # Преобразуем articles в словари
        articles_dict = [a if isinstance(a, dict) else a.dict() for a in found_articles]
        
        # Шаг 3: Верификация фактов
        async with httpx.AsyncClient(timeout=timeout) as client:
            verifier_response = await client.post(
                f"{VERIFIER_SERVICE_URL}/verify",
                json={
                    "claims": claims_dict,
                    "articles": articles_dict,
                    "original_text": request.text
                }
            )
            verifier_response.raise_for_status()
            verifier_data = verifier_response.json()
        
        # Шаг 4: Формирование результата
        claim_results = []
        for claim_data in verifier_data.get("claim_results", []):
            evidence_list = [
                ClaimEvidence(**ev) for ev in claim_data.get("evidence", [])
            ]
            claim_results.append(
                ClaimResult(
                    claim=claim_data["claim"],
                    label=claim_data["label"],
                    evidence=evidence_list
                )
            )
        
        sources = [
            SourceInfo(**src) for src in verifier_data.get("sources", [])
        ]
        
        # Определение общего статуса
        status = _determine_status(claim_results, verifier_data.get("confidence", 0.0))
        
        # Сохранение результата в БД (асинхронно)
        background_tasks.add_task(
            save_verification_result,
            request.text,
            status,
            verifier_data.get("confidence", 0.0),
            claim_results,
            sources
        )
        
        return VerifyResponse(
            status=status,
            confidence=verifier_data.get("confidence", 0.0),
            claims=claim_results,
            sources=sources,
            warnings=verifier_data.get("warnings", [])
        )
        
    except httpx.TimeoutException as e:
        import traceback
        error_detail = f"Таймаут при обращении к сервису: {str(e)}"
        print(f"Timeout error: {traceback.format_exc()}")
        raise HTTPException(status_code=504, detail=error_detail)
    except httpx.HTTPError as e:
        import traceback
        error_detail = f"Ошибка HTTP при обращении к сервису: {str(e)}"
        print(f"HTTP error: {traceback.format_exc()}")
        raise HTTPException(status_code=503, detail=error_detail)
    except Exception as e:
        import traceback
        error_detail = f"Ошибка обработки: {str(e)}"
        print(f"Unexpected error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)


def _determine_status(claim_results: List[ClaimResult], confidence: float) -> VerificationStatus:
    """Определение общего статуса на основе результатов по утверждениям"""
    if not claim_results:
        return VerificationStatus.INSUFFICIENT_DATA
    
    entails_count = sum(1 for c in claim_results if c.label == "ENTAILS")
    contradicts_count = sum(1 for c in claim_results if c.label == "CONTRADICTS")
    neutral_count = sum(1 for c in claim_results if c.label == "NEUTRAL")
    total = len(claim_results)
    
    entails_ratio = entails_count / total
    contradicts_ratio = contradicts_count / total
    
    # Если есть противоречия - статус REFUTED
    if contradicts_ratio > 0.3:
        return VerificationStatus.REFUTED
    
    # Если большинство подтверждено
    if entails_ratio >= 0.6:
        return VerificationStatus.CONFIRMED
    
    # Если часть подтверждена
    if entails_ratio > 0.2:
        return VerificationStatus.PARTIALLY_CONFIRMED
    
    # Недостаточно данных
    return VerificationStatus.INSUFFICIENT_DATA


async def save_verification_result(
    text: str,
    status: VerificationStatus,
    confidence: float,
    claims: List[ClaimResult],
    sources: List[SourceInfo]
):
    """Сохранение результата проверки в БД"""
    # TODO: Реализовать сохранение в PostgreSQL
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

