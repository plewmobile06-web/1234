# ระบบสนับสนุนการตัดสินใจทำนายความเสี่ยงพ้นสภาพนักศึกษา (DSS)

ระบบสำหรับอาจารย์ที่ปรึกษา ทำนายความเสี่ยงพ้นสภาพนักศึกษาด้วย XGBoost + อธิบายผลด้วย SHAP
ร่วมกับเกณฑ์ข้อบังคับการพ้นสภาพ (GPAX / วิทยาทัณฑ์) และให้ AI (Claude) อ่านภาพใบแสดงผลการเรียนได้

- Backend: FastAPI + SQLAlchemy + PostgreSQL
- Model: XGBoost (`train_model.py`, `ml_models/`) ฝึกจากข้อมูลจำลอง (synthetic) ใช้สาธิตระบบเท่านั้น
- Frontend: `index.html` (Tailwind)

## รันในเครื่อง

1. สร้างไฟล์ `.env`
   ```
   DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@localhost:5432/dropout_dss
   # ANTHROPIC_API_KEY=...   (ไม่บังคับ ใช้กับปุ่มให้ AI อ่านเอกสาร)
   ```
2. `pip install -r requirements.txt`
3. `python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload`
4. เปิด http://127.0.0.1:8000 (คู่มือ API: http://127.0.0.1:8000/docs)

## นำขึ้นเว็บ

ใช้ Render Blueprint ตาม `render.yaml` ดูขั้นตอนใน `วิธีนำระบบขึ้นเว็บ.txt`
