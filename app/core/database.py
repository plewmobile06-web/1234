import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# อ่านค่าจากไฟล์ .env (ในเครื่อง) หรือจาก Environment Variables ของ server บน cloud
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("ไม่พบ DATABASE_URL กรุณาตั้งค่าในไฟล์ .env หรือ Environment Variables")

# ผู้ให้บริการ cloud (เช่น Render) ให้ URL ขึ้นต้นด้วย postgres:// หรือ postgresql:// แต่ SQLAlchemy ต้องระบุ driver
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
