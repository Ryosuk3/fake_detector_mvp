from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from io import BytesIO
import os
import json
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta
import trafilatura
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np
from minio import Minio
from minio.error import S3Error
import hashlib
from simhash import Simhash
import feedparser
import httpx
from bs4 import BeautifulSoup
import re

app = FastAPI(title="Search Service", version="1.0.0")

# Конфигурация
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# Загрузка доверенных доменов
TRUSTED_DOMAINS = []
DOMAIN_WEIGHTS = {}
try:
    config_path = os.path.join(os.path.dirname(__file__), "config", "trusted_domains.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        TRUSTED_DOMAINS = config.get("domains", [])
        DOMAIN_WEIGHTS = config.get("domain_weights", {})
except FileNotFoundError:
    # Fallback конфигурация
    TRUSTED_DOMAINS = ["ria.ru", "tass.ru", "vedomosti.ru"]
    DOMAIN_WEIGHTS = {domain: 1.0 for domain in TRUSTED_DOMAINS}

# Инициализация клиентов
qdrant_client = QdrantClient(url=QDRANT_URL)
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Инициализация модели эмбеддингов (ленивая загрузка)
embedding_model = None

def get_embedding_model():
    """Ленивая загрузка модели эмбеддингов"""
    global embedding_model
    if embedding_model is None:
        print("Загрузка модели эмбеддингов...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return embedding_model

@app.on_event("startup")
async def startup_event():
    """Предзагрузка модели при старте сервиса"""
    print("Предзагрузка модели эмбеддингов при старте...")
    try:
        get_embedding_model()
        print("Модель эмбеддингов успешно загружена")
    except Exception as e:
        print(f"Ошибка при предзагрузке модели: {e}. Модель будет загружена при первом запросе.")

# Создание коллекции в Qdrant (если не существует)
COLLECTION_NAME = "news_articles"
try:
    qdrant_client.get_collection(COLLECTION_NAME)
    print(f"Коллекция {COLLECTION_NAME} уже существует")
except Exception as e:
    # Коллекция не существует, создаем
    try:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Коллекция {COLLECTION_NAME} успешно создана")
    except Exception as create_error:
        # Если коллекция уже существует (409), это нормально
        if "409" in str(create_error) or "already exists" in str(create_error).lower():
            print(f"Коллекция {COLLECTION_NAME} уже существует (это нормально)")
        else:
            print(f"Предупреждение: не удалось создать/проверить коллекцию: {create_error}")


class Claim(BaseModel):
    text: str
    importance: float = 1.0
    entities: List[str] = []


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int


class SearchRequest(BaseModel):
    claims: List[Claim]
    entities: List[Entity]
    text: str


class Article(BaseModel):
    url: str
    domain: str
    title: str
    date: Optional[str] = None
    text: str
    chunks: List[str] = []
    trust_level: float = 1.0
    embedding: Optional[List[float]] = None


class SearchResponse(BaseModel):
    articles: List[Article]
    total_found: int


def is_trusted_domain(url: str) -> bool:
    """Проверка, является ли домен доверенным"""
    try:
        domain = urlparse(url).netloc
        # Удаление www.
        domain = domain.replace("www.", "")
        return domain in TRUSTED_DOMAINS
    except Exception:
        return False


def get_domain_weight(domain: str) -> float:
    """Получение веса домена"""
    domain = domain.replace("www.", "")
    return DOMAIN_WEIGHTS.get(domain, 0.5)


# RSS фиды и главные страницы доверенных источников
TRUSTED_SOURCES_CONFIG = {
    "ria.ru": {
        "rss": ["https://ria.ru/export/rss2/index.xml"],
        "main_page": "https://ria.ru",
        "article_selector": "div.article__text"
    },
    "tass.ru": {
        "rss": ["https://tass.ru/rss/v2.xml"],
        "main_page": "https://tass.ru",
        "article_selector": "div.text-block"
    },
    "interfax.ru": {
        "rss": ["https://www.interfax.ru/rss.asp"],
        "main_page": "https://www.interfax.ru",
        "article_selector": "div.text"
    },
    "rt.com": {
        "rss": ["https://russian.rt.com/rss"],
        "main_page": "https://russian.rt.com",
        "article_selector": "div.article__text"
    },
    "lenta.ru": {
        "rss": ["https://lenta.ru/rss"],
        "main_page": "https://lenta.ru",
        "article_selector": "div.b-topic__body"
    },
    "kommersant.ru": {
        "rss": ["https://www.kommersant.ru/RSS/news.xml"],
        "main_page": "https://www.kommersant.ru",
        "article_selector": "div.article_text"
    },
    "vedomosti.ru": {
        "rss": ["https://www.vedomosti.ru/rss/news"],
        "main_page": "https://www.vedomosti.ru",
        "article_selector": "div.article-body"
    },
    "rbc.ru": {
        "rss": ["https://www.rbc.ru/rss/full"],
        "main_page": "https://www.rbc.ru",
        "article_selector": "div.article__text"
    },
    "gazeta.ru": {
        "rss": ["https://www.gazeta.ru/export/rss/lenta.xml"],
        "main_page": "https://www.gazeta.ru",
        "article_selector": "div.b_article-text"
    },
    "mk.ru": {
        "rss": ["https://www.mk.ru/rss/index.xml"],
        "main_page": "https://www.mk.ru",
        "article_selector": "div.article__body"
    }
}


def fetch_rss_articles(domain: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Получение статей из RSS фида"""
    articles = []
    config = TRUSTED_SOURCES_CONFIG.get(domain)
    
    if not config or not config.get("rss"):
        return articles
    
    try:
        for rss_url in config["rss"]:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:limit]:
                url = entry.get("link", "")
                if not url or not is_trusted_domain(url):
                    continue
                
                # Парсинг даты
                date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        date = datetime(*entry.published_parsed[:6]).isoformat()
                    except:
                        pass
                
                articles.append({
                    "url": url,
                    "title": entry.get("title", ""),
                    "date": date,
                    "summary": entry.get("summary", ""),
                    "domain": domain
                })
    except Exception as e:
        print(f"Ошибка при получении RSS для {domain}: {e}")
    
    return articles


def fetch_main_page_articles(domain: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Получение ссылок на статьи с главной страницы"""
    articles = []
    config = TRUSTED_SOURCES_CONFIG.get(domain)
    
    if not config or not config.get("main_page"):
        return articles
    
    try:
        response = httpx.get(config["main_page"], timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Поиск ссылок на статьи (универсальный подход)
        # Ищем ссылки, которые выглядят как статьи
        article_links = set()
        
        # Паттерны для разных сайтов
        patterns = {
            "ria.ru": [r"/\d{8}/", r"/\d{4}/\d{2}/\d{2}/"],
            "tass.ru": [r"/[a-z-]+/\d+", r"/\d{8}/"],
            "interfax.ru": [r"/news/\d+", r"/\d{4}/\d{2}/\d{2}/"],
            "rt.com": [r"/\d+", r"/news/\d+"],
            "lenta.ru": [r"/news/\d{4}/\d{2}/\d{2}/", r"/\d+"],
            "kommersant.ru": [r"/doc/\d+", r"/\d{4}/\d{2}/\d{2}/"],
            "vedomosti.ru": [r"/articles/\d{4}/\d{2}/\d{2}/", r"/news/\d+"],
            "rbc.ru": [r"/news/\d+", r"/\d{4}/\d{2}/\d{2}/"],
            "gazeta.ru": [r"/\d+", r"/news/\d+"],
            "mk.ru": [r"/\d+", r"/news/\d+"]
        }
        
        domain_patterns = patterns.get(domain, [r"/\d+", r"/article", r"/news"])
        
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            full_url = urljoin(config["main_page"], href)
            
            # Проверяем что это ссылка на статью
            if any(re.search(pattern, href) for pattern in domain_patterns):
                if is_trusted_domain(full_url) and full_url not in article_links:
                    article_links.add(full_url)
                    if len(article_links) >= limit:
                        break
        
        # Преобразуем в список словарей
        for url in list(article_links)[:limit]:
            articles.append({
                "url": url,
                "title": "",
                "date": None,
                "summary": "",
                "domain": domain
            })
            
    except Exception as e:
        print(f"Ошибка при парсинге главной страницы {domain}: {e}")
    
    return articles


def search_articles_web(claims: List[Claim], entities: List[Entity], limit_per_source: int = 10) -> List[str]:
    """
    Поиск статей в интернете через RSS фиды и парсинг главных страниц.
    
    Возвращает список URL статей из доверенных источников.
    """
    found_urls = []
    
    # Собираем ключевые слова для фильтрации
    keywords = set()
    for claim in claims:
        # Извлекаем существительные и важные слова
        words = claim.text.lower().split()
        keywords.update([w for w in words if len(w) > 4])
    
    for entity in entities:
        keywords.add(entity.text.lower())
    
    # Для каждого доверенного источника
    for domain in TRUSTED_DOMAINS:
        if domain not in TRUSTED_SOURCES_CONFIG:
            continue
        
        # Получаем статьи из RSS
        rss_articles = fetch_rss_articles(domain, limit=limit_per_source)
        
        # Получаем статьи с главной страницы
        main_page_articles = fetch_main_page_articles(domain, limit=limit_per_source)
        
        # Объединяем и фильтруем по релевантности
        all_articles = rss_articles + main_page_articles
        
        # Простая фильтрация по ключевым словам (если есть)
        if keywords:
            filtered_articles = []
            for article in all_articles:
                title_lower = article.get("title", "").lower()
                summary_lower = article.get("summary", "").lower()
                text_lower = title_lower + " " + summary_lower
                
                # Если есть совпадение ключевых слов
                if any(kw in text_lower for kw in keywords):
                    filtered_articles.append(article)
            
            # Если после фильтрации ничего не осталось, берем все
            if not filtered_articles:
                filtered_articles = all_articles[:limit_per_source]
        else:
            filtered_articles = all_articles[:limit_per_source]
        
        # Добавляем URL
        for article in filtered_articles:
            url = article.get("url")
            if url and url not in found_urls:
                found_urls.append(url)
    
    return found_urls[:limit_per_source * len(TRUSTED_DOMAINS)]


def parse_article(url: str) -> Optional[Dict[str, Any]]:
    """Парсинг статьи с веб-страницы"""
    try:
        # Используем trafilatura для извлечения контента
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        
        article = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_links=False,
            include_images=False,
            include_tables=False
        )
        
        if not article:
            return None
        
        # Извлечение метаданных
        metadata = trafilatura.extract_metadata(downloaded)
        
        domain = urlparse(url).netloc.replace("www.", "")
        
        return {
            "url": url,
            "domain": domain,
            "title": metadata.title if metadata else "",
            "date": metadata.date if metadata else None,
            "text": article,
            "trust_level": get_domain_weight(domain)
        }
    except Exception as e:
        print(f"Ошибка парсинга {url}: {e}")
        return None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Разбивка текста на чанки"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def compute_simhash(text: str) -> str:
    """Вычисление SimHash для дедупликации"""
    return str(Simhash(text).value)


def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Дедупликация статей по SimHash"""
    seen_hashes = set()
    unique_articles = []
    
    for article in articles:
        text = article.get("text", "")
        if not text:
            continue
        
        simhash = compute_simhash(text)
        if simhash not in seen_hashes:
            seen_hashes.add(simhash)
            unique_articles.append(article)
    
    return unique_articles


def index_article(article: Dict[str, Any], chunks: List[str]):
    """Индексация статьи в Qdrant"""
    try:
        # Генерация эмбеддингов для чанков
        model = get_embedding_model()
        chunk_embeddings = model.encode(chunks)
        
        # Сохранение в MinIO
        article_id = hashlib.md5(article["url"].encode()).hexdigest()
        bucket_name = "articles"
        
        # Создание bucket если не существует
        try:
            minio_client.make_bucket(bucket_name)
        except S3Error:
            pass
        
        # Сохранение статьи
        from io import BytesIO
        article_data = json.dumps(article, ensure_ascii=False, default=str).encode('utf-8')
        minio_client.put_object(
            bucket_name,
            f"{article_id}.json",
            data=BytesIO(article_data),
            length=len(article_data)
        )
        
        # Индексация чанков в Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            point_id = int(hashlib.md5(f"{article_id}_{i}".encode()).hexdigest()[:8], 16)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "url": article["url"],
                        "domain": article["domain"],
                        "title": article.get("title", ""),
                        "date": article.get("date"),
                        "chunk": chunk,
                        "chunk_index": i,
                        "trust_level": article.get("trust_level", 1.0)
                    }
                )
            )
        
        if points:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        
    except Exception as e:
        print(f"Ошибка индексации: {e}")


def search_relevant_articles(claims: List[Claim], limit: int = 20) -> List[Dict[str, Any]]:
    """Поиск релевантных статей в Qdrant"""
    if not claims:
        return []
    
    # Генерация эмбеддингов для claims
    model = get_embedding_model()
    claim_texts = [claim.text for claim in claims]
    claim_embeddings = model.encode(claim_texts)
    
    # Поиск в Qdrant
    all_results = []
    for embedding in claim_embeddings:
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding.tolist(),
            limit=limit,
            score_threshold=0.5
        )
        all_results.extend(results)
    
    # Группировка по URL и удаление дубликатов
    url_to_article = {}
    for result in all_results:
        url = result.payload.get("url")
        if url and url not in url_to_article:
            url_to_article[url] = {
                "url": url,
                "domain": result.payload.get("domain", ""),
                "title": result.payload.get("title", ""),
                "date": result.payload.get("date"),
                "trust_level": result.payload.get("trust_level", 1.0),
                "score": result.score
            }
    
    # Сортировка по релевантности и trust_level
    articles = sorted(
        url_to_article.values(),
        key=lambda x: (x["score"] * x["trust_level"]),
        reverse=True
    )
    
    return articles[:limit]


@app.get("/health")
async def health():
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        return {"status": "healthy", "service": "search_service"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/search", response_model=SearchResponse)
async def search_articles(request: SearchRequest):
    """
    Поиск релевантных статей для проверки утверждений.
    
    Процесс:
    1. Поиск в Qdrant по эмбеддингам claims
    2. Если недостаточно статей - поиск в интернете через RSS и парсинг
    3. Парсинг найденных статей
    4. Индексация новых статей в Qdrant
    """
    try:
        # Шаг 1: Поиск в индексе
        found_articles = search_relevant_articles(request.claims, limit=20)
        
        # Шаг 2: Если в индексе недостаточно статей, ищем в интернете
        if len(found_articles) < 5:
            print(f"Найдено только {len(found_articles)} статей в индексе, ищем в интернете...")
            web_urls = search_articles_web(request.claims, request.entities, limit_per_source=5)
            
            # Парсим найденные статьи
            for url in web_urls:
                # Проверяем, нет ли уже в индексе
                article_id = hashlib.md5(url.encode()).hexdigest()
                bucket_name = "articles"
                
                try:
                    # Проверяем наличие в MinIO
                    minio_client.stat_object(bucket_name, f"{article_id}.json")
                    # Уже есть, пропускаем
                    continue
                except Exception:
                    # Нет в хранилище, парсим
                    parsed = parse_article(url)
                    if parsed:
                        chunks = chunk_text(parsed["text"])
                        parsed["chunks"] = chunks
                        # Индексируем новую статью
                        index_article(parsed, chunks)
                        
                        # Добавляем в результаты
                        found_articles.append({
                            "url": parsed["url"],
                            "domain": parsed["domain"],
                            "title": parsed.get("title", ""),
                            "date": parsed.get("date"),
                            "trust_level": parsed.get("trust_level", 1.0),
                            "score": 0.7  # Средний score для новых статей
                        })
        
        # Шаг 3: Загрузка полных текстов статей из MinIO или парсинг
        articles_data = []
        for article_info in found_articles:
            url = article_info["url"]
            
            # Попытка загрузить из MinIO
            article_id = hashlib.md5(url.encode()).hexdigest()
            bucket_name = "articles"
            
            try:
                response = minio_client.get_object(bucket_name, f"{article_id}.json")
                article_data = json.loads(response.read().decode('utf-8'))
                article_data["chunks"] = chunk_text(article_data.get("text", ""))
                articles_data.append(Article(**article_data))
            except Exception:
                # Если нет в хранилище, парсим
                parsed = parse_article(url)
                if parsed:
                    chunks = chunk_text(parsed["text"])
                    parsed["chunks"] = chunks
                    articles_data.append(Article(**parsed))
                    # Индексируем новую статью
                    index_article(parsed, chunks)
        
        # Дедупликация
        unique_articles = deduplicate_articles([a.dict() for a in articles_data])
        articles_data = [Article(**a) for a in unique_articles]
        
        return SearchResponse(
            articles=articles_data,
            total_found=len(articles_data)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

