import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

# 1. จำลองชุดข้อมูลนักศึกษา (Synthetic Dataset) สำหรับฝึกสอนโมเดล
# ใช้ตัวแปรวิกฤต (Critical Features) จากผลการเรียนเท่านั้น
np.random.seed(42)
n_samples = 60000

data = {
    'gpax': np.random.uniform(0.0, 4.0, n_samples),                     # GPAX ล่าสุด
    'credit_completion_rate': np.random.uniform(30.0, 120.0, n_samples),  # หน่วยกิตสะสม / หน่วยกิตตามแผน (%)
    'fw_count': np.random.randint(0, 9, n_samples),                     # จำนวนวิชา F/W ในเทอมล่าสุด
    'probation_terms': np.random.randint(0, 7, n_samples),              # จำนวนภาคเรียนที่ติดวิทยาทัณฑ์ติดต่อกัน
}

df = pd.DataFrame(data)

# กำหนดความเสี่ยงพ้นสภาพแบบค่อยเป็นค่อยไป (logistic) แทนกฎตัดขาด
# เพื่อให้ความเสี่ยงลดลงทีละน้อยเมื่อเกรดสูงขึ้น และมีกลุ่มเสี่ยงปานกลางเกิดขึ้นได้
risk_logit = (
    1.8 * (2.0 - df['gpax'])
    + 0.035 * (85 - df['credit_completion_rate'])
    + 0.4 * (df['fw_count'] - 1)
    + 0.45 * df['probation_terms']
    - 0.6
)
risk_prob = 1 / (1 + np.exp(-risk_logit))

# สุ่ม Target จากความน่าจะเป็น (1 = พ้นสภาพ, 0 = ปกติ) โมเดลจึงเรียนรู้ระดับความเสี่ยง ไม่ใช่แค่ใช่/ไม่ใช่
df['dropout_risk'] = (np.random.uniform(0, 1, n_samples) < risk_prob).astype(int)

X = df.drop(columns=['dropout_risk'])
y = df['dropout_risk']

# 2. ฝึกสอนโมเดล XGBoost (ต้นไม้ตื้น + learning rate ต่ำ ให้ผลทำนายเรียบขึ้น)
# monotone_constraints: เกรด/หน่วยกิตสะสมสูงขึ้น -> ความเสี่ยงต้องไม่เพิ่ม
#                       วิชา F/W และภาคเรียนที่ติดวิทยาทัณฑ์มากขึ้น -> ความเสี่ยงต้องไม่ลด
# ลำดับ: gpax, credit_completion_rate, fw_count, probation_terms
model = xgb.XGBClassifier(
    n_estimators=800, max_depth=2, learning_rate=0.03,
    min_child_weight=20, subsample=0.8,
    monotone_constraints=(-1, -1, 1, 1),
    random_state=42
)
model.fit(X, y)

# 3. บันทึกโมเดลลงในโฟลเดอร์ ml_models/
os.makedirs("ml_models", exist_ok=True)
model_path = "ml_models/xgb_dropout_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"ฝึกสอนโมเดลสำเร็จและบันทึกไว้ที่: {model_path}")
