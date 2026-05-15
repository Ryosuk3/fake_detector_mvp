from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON
from datetime import datetime
import asyncio
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@postgres:5432/fake_detector"
)

# Для синхронных операций (alembic)
SYNC_DATABASE_URL = DATABASE_URL.replace("+asyncpg", "")

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


class VerificationRequest(Base):
    __tablename__ = "verification_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    status = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    result = Column(JSON, nullable=True)


class Source(Base):
    __tablename__ = "sources"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    domain = Column(String, index=True)
    title = Column(String)
    date = Column(DateTime)
    trust_level = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Инициализация БД - создание таблиц"""
    retries = int(os.getenv("DB_INIT_RETRIES", "12"))
    delay = float(os.getenv("DB_INIT_RETRY_DELAY_SEC", "2"))
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            return
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"Database is not ready yet, retrying init_db "
                f"({attempt}/{retries}): {exc}",
                flush=True,
            )
            await asyncio.sleep(delay)

    raise last_error
