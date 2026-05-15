from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple, Set
import os
import threading
import logging
from urllib.parse import urlparse

from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import snapshot_download as hf_snapshot_download

app = FastAPI(title="Verifier Service", version="1.0.0")

# Конфигурация
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "auto")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
NLI_BATCH_SIZE = int(os.getenv("NLI_BATCH_SIZE", "8"))
NLI_MODEL_NAME = os.getenv(
    "NLI_MODEL_NAME",
    "cointegrated/rubert-base-cased-nli-threeway"
)
NLI_MAX_LENGTH = int(os.getenv("NLI_MAX_LENGTH", "384"))
MAX_ARTICLES_TO_VERIFY = int(os.getenv("MAX_ARTICLES_TO_VERIFY", "8"))
MAX_CHUNKS_PER_ARTICLE = int(os.getenv("MAX_CHUNKS_PER_ARTICLE", "12"))
EVIDENCE_SIMILARITY_MIN_SCORE = float(os.getenv("EVIDENCE_SIMILARITY_MIN_SCORE", "0.45"))
EVIDENCE_RELEVANCE_MIN_SCORE = float(os.getenv("EVIDENCE_RELEVANCE_MIN_SCORE", "0.55"))
CLAIM_LABEL_MIN_CONFIDENCE = float(os.getenv("CLAIM_LABEL_MIN_CONFIDENCE", "0.65"))
CONTRADICTION_MIN_MARGIN = float(os.getenv("CONTRADICTION_MIN_MARGIN", "0.08"))
STATUS_REFUTED_MIN_CONTRADICT_RATIO = float(os.getenv("STATUS_REFUTED_MIN_CONTRADICT_RATIO", "0.6"))
STATUS_REFUTED_MIN_CONTRADICT_CLAIMS = int(os.getenv("STATUS_REFUTED_MIN_CONTRADICT_CLAIMS", "2"))
STATUS_REFUTED_MIN_UNIQUE_DOMAINS = int(os.getenv("STATUS_REFUTED_MIN_UNIQUE_DOMAINS", "2"))
STATUS_CONFIRMED_MIN_ENTAIL_RATIO = float(os.getenv("STATUS_CONFIRMED_MIN_ENTAIL_RATIO", "0.6"))
STATUS_PARTIAL_MIN_ENTAIL_RATIO = float(os.getenv("STATUS_PARTIAL_MIN_ENTAIL_RATIO", "0.0"))
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() in {"1", "true", "yes"}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("verifier_service")

# Ленивая загрузка моделей
embedding_model = None
nli_model = None
nli_tokenizer = None
nli_label_mapping = None
model_load_error = None
model_init_lock = threading.Lock()


def get_torch_device() -> str:
    """Определение устройства для инференса"""
    if TORCH_DEVICE != "auto":
        return TORCH_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


EMBEDDING_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def download_snapshot(repo_id: str) -> str:
    """Load an HF snapshot from local cache first; fall back to network if needed."""
    try:
        return hf_snapshot_download(
            repo_id,
            ignore_patterns=["*.onnx", "*.onnx_data", "onnx/*"],
            local_files_only=True,
        )
    except Exception:
        return hf_snapshot_download(
            repo_id,
            ignore_patterns=["*.onnx", "*.onnx_data", "onnx/*"],
            resume_download=True,
        )


def get_embedding_model():
    """Ленивая загрузка модели эмбеддингов"""
    global embedding_model, model_load_error
    if embedding_model is None:
        with model_init_lock:
            if embedding_model is None:
                print("Загрузка модели эмбеддингов для верификации...")
                try:
                    local_path = download_snapshot(EMBEDDING_REPO)
                    embedding_model = SentenceTransformer(
                        local_path,
                        device=get_torch_device()
                    )
                    model_load_error = None
                except Exception as exc:
                    model_load_error = f"embedding_model: {exc}"
                    logger.exception("Не удалось загрузить embedding-модель")
                    raise
    return embedding_model


def get_nli_components():
    """Ленивая загрузка NLI модели и токенизатора"""
    global nli_model, nli_tokenizer, nli_label_mapping, model_load_error

    with model_init_lock:
        if nli_model is None or nli_tokenizer is None:
            print(f"Загрузка NLI модели {NLI_MODEL_NAME}...")
            try:
                try:
                    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, local_files_only=True)
                    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME, local_files_only=True)
                except Exception:
                    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
                    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
                nli_model.to(get_torch_device())
                nli_model.eval()
                model_load_error = None
            except Exception as exc:
                model_load_error = f"nli_model: {exc}"
                logger.exception("Не удалось загрузить NLI-модель")
                raise

        if nli_label_mapping is None:
            nli_label_mapping = build_label_mapping(nli_model.config.id2label)

    return nli_model, nli_tokenizer, nli_label_mapping


@app.on_event("startup")
async def startup_event():
    if PRELOAD_MODELS:
        try:
            get_embedding_model()
            get_nli_components()
        except Exception:
            logger.exception("Модели не были прогреты на старте; сервис продолжит работу")


def build_label_mapping(id2label: Dict[int, str]) -> Dict[str, int]:
    """Определение индексов entailment/contradiction/neutral по строковым label"""
    mapping: Dict[str, int] = {}

    for index, label in id2label.items():
        normalized_label = str(label).lower()
        if "entail" in normalized_label:
            mapping["entailment"] = int(index)
        elif "contrad" in normalized_label:
            mapping["contradiction"] = int(index)
        elif "neutral" in normalized_label:
            mapping["neutral"] = int(index)

    required_labels = {"entailment", "contradiction", "neutral"}
    if not required_labels.issubset(mapping):
        raise RuntimeError(f"Не удалось определить label mapping NLI модели: {id2label}")

    return mapping


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
    evidence_status: str = "FOUND"  # FOUND | NOT_FOUND
    drop_reason: Optional[str] = None


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
    embeddings = model.encode(
        [text1, text2],
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_tensor=True
    )
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return float(similarity)


def score_evidence_chunks(claim: str, chunks: List[str]) -> List[Tuple[str, float]]:
    """Батчевое ранжирование чанков статьи по сходству с claim"""
    if not chunks:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        [claim] + chunks,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_tensor=True
    )
    similarity_scores = util.cos_sim(embeddings[0], embeddings[1:])[0].detach().cpu().tolist()
    return list(zip(chunks, similarity_scores))


def classify_entailment(claim: str, evidence_text: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Классификация отношения между claim и evidence.
    Возвращает (label, confidence, raw_scores).
    """
    return classify_entailment_batch(claim, [evidence_text])[0]


def classify_entailment_batch(
    claim: str,
    evidence_texts: List[str]
) -> List[Tuple[str, float, Dict[str, float]]]:
    """Батчевая NLI-классификация claim против нескольких evidence"""
    if not evidence_texts:
        return []

    model, tokenizer, label_mapping = get_nli_components()
    results = []

    for batch_start in range(0, len(evidence_texts), NLI_BATCH_SIZE):
        batch_texts = evidence_texts[batch_start:batch_start + NLI_BATCH_SIZE]
        model_inputs = tokenizer(
            batch_texts,
            [claim] * len(batch_texts),
            truncation=True,
            padding=True,
            max_length=NLI_MAX_LENGTH,
            return_tensors="pt"
        )
        model_inputs = {key: value.to(get_torch_device()) for key, value in model_inputs.items()}

        with torch.no_grad():
            logits = model(**model_inputs).logits
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        for probability_vector in probabilities:
            entailment_score = float(probability_vector[label_mapping["entailment"]])
            contradiction_score = float(probability_vector[label_mapping["contradiction"]])
            neutral_score = float(probability_vector[label_mapping["neutral"]])
            raw_scores = {
                "entailment": entailment_score,
                "contradiction": contradiction_score,
                "neutral": neutral_score
            }

            if entailment_score >= contradiction_score and entailment_score >= neutral_score:
                results.append(("ENTAILS", entailment_score, raw_scores))
            elif contradiction_score >= entailment_score and contradiction_score >= neutral_score:
                results.append(("CONTRADICTS", contradiction_score, raw_scores))
            else:
                results.append(("NEUTRAL", neutral_score, raw_scores))

    return results


def find_best_evidence(claim: str, article: Article, top_k: int = 3) -> List[Evidence]:
    """Поиск лучших фрагментов статьи для подтверждения claim"""
    if not article.chunks:
        words = article.text.split()
        chunk_size = 100
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    else:
        chunks = article.chunks

    chunks = chunks[:MAX_CHUNKS_PER_ARTICLE]
    chunk_scores = score_evidence_chunks(claim, chunks)
    chunk_scores.sort(key=lambda x: x[1], reverse=True)

    evidence_list = []
    for chunk, score in chunk_scores[:top_k]:
        if score >= EVIDENCE_SIMILARITY_MIN_SCORE:
            evidence_list.append(
                Evidence(
                    url=article.url,
                    title=article.title,
                    date=article.date,
                    snippet=chunk[:240] + "..." if len(chunk) > 240 else chunk
                )
            )

    return evidence_list


def verify_claims(claims: List[Claim], articles: List[Article]) -> Dict[str, Any]:
    """Верификация всех claims против статей"""
    claim_results = []
    all_sources = {}

    ranked_articles = sorted(articles, key=lambda article: article.trust_level, reverse=True)
    candidate_articles = ranked_articles[:MAX_ARTICLES_TO_VERIFY]

    for claim in claims:
        best_entails_score = 0.0
        best_contradicts_score = 0.0
        best_relevance_score = 0.0
        best_entails_evidence: List[Evidence] = []
        best_contradicts_evidence: List[Evidence] = []
        best_relevant_evidence: List[Evidence] = []
        article_evidence_candidates: List[Tuple[Article, List[Evidence]]] = []

        for article in candidate_articles:
            evidence_list = find_best_evidence(claim.text, article, top_k=2)
            if not evidence_list:
                continue
            article_evidence_candidates.append((article, evidence_list))

            all_sources[article.url] = {
                "url": article.url,
                "domain": article.domain,
                "date": article.date,
                "trust_level": article.trust_level
            }

        candidate_snippets = [
            evidence_list[0].snippet
            for _, evidence_list in article_evidence_candidates
        ]
        if not candidate_snippets:
            drop_reason = "no_candidate_above_similarity_threshold"
            logger.info("claim='%s' drop_reason=%s", claim.text[:160], drop_reason)
            claim_results.append(
                ClaimResult(
                    claim=claim.text,
                    label="NEUTRAL",
                    confidence=0.3,
                    evidence=[],
                    evidence_status="NOT_FOUND",
                    drop_reason=drop_reason,
                )
            )
            continue

        candidate_predictions = classify_entailment_batch(claim.text, candidate_snippets)

        for (article, evidence_list), (label, confidence, raw_scores) in zip(article_evidence_candidates, candidate_predictions):
            relevance_score = max(raw_scores.get("entailment", 0.0), raw_scores.get("contradiction", 0.0))
            if relevance_score < EVIDENCE_RELEVANCE_MIN_SCORE:
                continue

            if relevance_score > best_relevance_score:
                best_relevance_score = relevance_score
                best_relevant_evidence = evidence_list[:2]

            if label == "ENTAILS" and confidence > best_entails_score:
                best_entails_score = confidence
                best_entails_evidence = evidence_list[:2]
            elif label == "CONTRADICTS" and confidence > best_contradicts_score:
                best_contradicts_score = confidence
                best_contradicts_evidence = evidence_list[:2]

        if best_relevance_score < EVIDENCE_RELEVANCE_MIN_SCORE or not best_relevant_evidence:
            drop_reason = "no_snippet_passed_relevance_threshold"
            logger.info("claim='%s' drop_reason=%s", claim.text[:160], drop_reason)
            claim_results.append(
                ClaimResult(
                    claim=claim.text,
                    label="NEUTRAL",
                    confidence=0.3,
                    evidence=[],
                    evidence_status="NOT_FOUND",
                    drop_reason=drop_reason,
                )
            )
            continue

        if best_contradicts_score >= CLAIM_LABEL_MIN_CONFIDENCE and best_contradicts_score >= (best_entails_score + CONTRADICTION_MIN_MARGIN):
            final_label = "CONTRADICTS"
            final_confidence = best_contradicts_score
            final_evidence = best_contradicts_evidence
            drop_reason = None
        elif best_entails_score >= CLAIM_LABEL_MIN_CONFIDENCE:
            final_label = "ENTAILS"
            final_confidence = best_entails_score
            final_evidence = best_entails_evidence
            drop_reason = None
        else:
            final_label = "NEUTRAL"
            final_confidence = max(best_entails_score, best_contradicts_score, 0.3)
            final_evidence = best_relevant_evidence
            drop_reason = "no_label_passed_claim_confidence_threshold"

        claim_results.append(
            ClaimResult(
                claim=claim.text,
                label=final_label,
                confidence=final_confidence,
                evidence=final_evidence if final_evidence else [],
                evidence_status="FOUND",
                drop_reason=drop_reason,
            )
        )

    return {
        "claim_results": [claim_result.dict() for claim_result in claim_results],
        "sources": list(all_sources.values())
    }


def compute_overall_confidence(claim_results: List[ClaimResult]) -> float:
    """Вычисление общей уверенности на основе результатов по claims"""
    if not claim_results:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for claim_result in claim_results:
        if claim_result.label == "ENTAILS":
            weight = 1.0
        elif claim_result.label == "CONTRADICTS":
            weight = 0.5
        else:
            weight = 0.3

        weighted_sum += claim_result.confidence * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return min(weighted_sum / total_weight, 1.0)


def _extract_domains_from_evidence(evidence_items: List[Evidence]) -> Set[str]:
    domains: Set[str] = set()
    for evidence in evidence_items:
        url = (evidence.url or "").strip()
        if not url:
            continue
        try:
            domain = urlparse(url).netloc.replace("www.", "").lower()
        except Exception:
            domain = ""
        if domain:
            domains.add(domain)
    return domains


def _should_mark_refuted(claim_results: List[ClaimResult]) -> bool:
    if not claim_results:
        return False
    contradict_results = [claim_result for claim_result in claim_results if claim_result.label == "CONTRADICTS"]
    contradict_count = len(contradict_results)
    contradict_ratio = contradict_count / len(claim_results)
    if contradict_count < STATUS_REFUTED_MIN_CONTRADICT_CLAIMS:
        return False
    if contradict_ratio <= STATUS_REFUTED_MIN_CONTRADICT_RATIO:
        return False
    contradict_domains: Set[str] = set()
    for claim_result in contradict_results:
        contradict_domains.update(_extract_domains_from_evidence(claim_result.evidence))
    return len(contradict_domains) >= STATUS_REFUTED_MIN_UNIQUE_DOMAINS


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "verifier_service",
        "device": get_torch_device(),
        "cuda_available": torch.cuda.is_available(),
        "nli_model": NLI_MODEL_NAME,
        "preload_models": PRELOAD_MODELS,
        "embedding_model_loaded": embedding_model is not None,
        "nli_model_loaded": nli_model is not None and nli_tokenizer is not None,
        "model_load_error": model_load_error,
        "label_min_confidence": CLAIM_LABEL_MIN_CONFIDENCE,
        "contradiction_min_margin": CONTRADICTION_MIN_MARGIN,
        "evidence_similarity_min_score": EVIDENCE_SIMILARITY_MIN_SCORE,
        "evidence_relevance_min_score": EVIDENCE_RELEVANCE_MIN_SCORE,
        "status_refuted_min_contradict_ratio": STATUS_REFUTED_MIN_CONTRADICT_RATIO,
        "status_refuted_min_contradict_claims": STATUS_REFUTED_MIN_CONTRADICT_CLAIMS,
        "status_refuted_min_unique_domains": STATUS_REFUTED_MIN_UNIQUE_DOMAINS,
        "status_confirmed_min_entail_ratio": STATUS_CONFIRMED_MIN_ENTAIL_RATIO,
        "status_partial_min_entail_ratio": STATUS_PARTIAL_MIN_ENTAIL_RATIO,
    }


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

        results = verify_claims(request.claims, request.articles)

        claim_results = [ClaimResult(**claim_result) for claim_result in results["claim_results"]]
        sources = [SourceInfo(**source) for source in results["sources"]]
        overall_confidence = compute_overall_confidence(claim_results)

        entails_count = sum(1 for claim_result in claim_results if claim_result.label == "ENTAILS")
        contradicts_count = sum(1 for claim_result in claim_results if claim_result.label == "CONTRADICTS")
        total = len(claim_results)

        if _should_mark_refuted(claim_results):
            status = "REFUTED"
        elif entails_count >= total * STATUS_CONFIRMED_MIN_ENTAIL_RATIO:
            status = "CONFIRMED"
        elif entails_count > total * STATUS_PARTIAL_MIN_ENTAIL_RATIO:
            status = "PARTIALLY_CONFIRMED"
        else:
            status = "INSUFFICIENT_DATA"

        warnings = []
        if status == "INSUFFICIENT_DATA":
            warnings.append("Недостаточно данных для проверки утверждений")
        elif status == "PARTIALLY_CONFIRMED":
            warnings.append("Часть утверждений не подтверждена")

        if contradicts_count > 0 and status != "REFUTED":
            warnings.append("Contradicting snippets exist, but not enough for global refutation")

        not_found_count = sum(
            1 for claim_result in claim_results if claim_result.evidence_status == "NOT_FOUND"
        )
        if not_found_count > 0:
            warnings.append(f"No relevant evidence found for {not_found_count} of {total} claims")

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
