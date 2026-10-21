import pickle
import pandas as pd
import shap
import traceback
import io
from typing import List
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.models.schemas import ข้อมูลนักศึกษาขาเข้า, ผลการทำนายความเสี่ยง, ปัจจัยความเสี่ยงจากAI
from app.core.database import get_db
from app.models.database_models import Student, RiskPrediction

router = APIRouter(prefix="/api/v1", tags=["ระบบจัดการและทำนายความเสี่ยงนักศึกษา"])

# โหลดโมเดล AI
model_path = "ml_models/xgb_dropout_model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
except Exception as e:
    model = None
    explainer = None
    print("⚠️ ไม่สามารถโหลดโมเดล AI ได้:", e)

# เกณฑ์ตามข้อบังคับการพ้นสภาพ (ค่าทั่วไปของมหาวิทยาลัยไทย ควรตรวจสอบกับข้อบังคับของมหาวิทยาลัยก่อนใช้งานจริง)
GPAX_DISMISS = 1.50          # GPAX ต่ำกว่านี้ = พ้นสภาพ
GPAX_MIN_GRADUATE = 2.00     # GPAX ต่ำกว่านี้ = ติดวิทยาทัณฑ์ / ยังจบการศึกษาไม่ได้
MAX_PROBATION_TERMS = 2      # ติดวิทยาทัณฑ์ติดต่อกันได้ไม่เกินกี่ภาคเรียน

# เกณฑ์ระดับความเสี่ยงจากคะแนน AI: สีแดง 51 ขึ้นไป | สีเหลือง 41 - 50 | สีเขียว 40 ลงไป
RISK_LEVELS = [
    "🟢 กลุ่มปกติ (ไม่มีความเสี่ยง)",
    "🟡 กลุ่มเสี่ยงปานกลาง (ควรเฝ้าระวัง)",
    "🔴 กลุ่มเสี่ยงสูง (ต้องรีบดูแลด่วน)",
]

thai_names = {
    'gpax': 'เกรดเฉลี่ยสะสม (GPAX)',
    'credit_completion_rate': 'หน่วยกิตสะสมเทียบแผนการเรียน (%)',
    'fw_count': 'จำนวนวิชา F/W เทอมล่าสุด',
    'midterm_score': 'คะแนนเก็บ/กลางภาค (%)',
    'probation_terms': 'จำนวนภาคเรียนที่ติดวิทยาทัณฑ์',
}

MAX_TERMS = 8

# คอลัมน์ของไฟล์ Excel แบบเทมเพลต (ตรงกับข้อมูลที่กรอกในฟอร์ม)
# ผลการเรียนรายภาคใช้คอลัมน์ term1_gpax, term1_fw, term1_midterm ... term8_* (ต้องมีอย่างน้อยภาคที่ 1)
TEMPLATE_COLUMNS = [
    'student_id', 'first_name', 'last_name', 'academic_year',
    'credits_earned', 'credits_planned'
]
TERM_COLUMNS = [col for t in range(1, MAX_TERMS + 1)
                for col in (f'term{t}_gpax', f'term{t}_fw', f'term{t}_midterm')]


def นับภาคเรียนวิทยาทัณฑ์(term_results):
    """นับจำนวนภาคเรียนล่าสุดที่ GPAX ต่ำกว่า 2.00 ติดต่อกัน (นับรวมภาคล่าสุด)"""
    count = 0
    for term in reversed(term_results):
        if term['gpax'] < GPAX_MIN_GRADUATE:
            count += 1
        else:
            break
    return count


def สร้างฟีเจอร์(term_results, credits_earned, credits_planned):
    """term_results: list ของ {'gpax', 'fw_count', 'midterm_score'} เรียงจากภาคแรกถึงภาคล่าสุด
    ใช้ GPAX, วิชา F/W และคะแนนเก็บของภาคล่าสุด"""
    latest = term_results[-1]
    # หน่วยกิตที่เรียนเร็วกว่าแผนมากไม่ได้ช่วยลดความเสี่ยงเพิ่ม จึงจำกัดไว้ที่ 120%
    credit_rate = min(credits_earned / credits_planned * 100, 120.0) if credits_planned > 0 else 100.0
    return {
        'gpax': float(latest['gpax']),
        'credit_completion_rate': round(credit_rate, 2),
        'fw_count': int(latest['fw_count']),
        'midterm_score': float(latest['midterm_score']),
        'probation_terms': นับภาคเรียนวิทยาทัณฑ์(term_results),
    }


def ตรวจเกณฑ์ข้อบังคับ(gpax, probation_terms, academic_year, n_terms):
    """คืนค่า (ระดับความเสี่ยงขั้นต่ำตามข้อบังคับ 0/1/2, รายการแจ้งเตือน)"""
    alerts = []
    level = 0
    if gpax < GPAX_DISMISS:
        level = 2
        if n_terms >= 2:
            alerts.append(f"GPAX {gpax:.2f} ต่ำกว่า {GPAX_DISMISS:.2f} หลังเรียนครบ {n_terms} ภาคเรียน เข้าเกณฑ์พ้นสภาพนักศึกษา")
        else:
            # เกณฑ์ 1.50 มักเริ่มใช้หลังผ่าน 2 ภาคเรียนแรก
            alerts.append(f"GPAX {gpax:.2f} ต่ำกว่า {GPAX_DISMISS:.2f} ถ้าสิ้นภาคเรียนที่ 2 ยังต่ำกว่าเกณฑ์จะพ้นสภาพนักศึกษา")
    elif gpax < GPAX_MIN_GRADUATE:
        if probation_terms > MAX_PROBATION_TERMS:
            level = 2
            alerts.append(
                f"GPAX ต่ำกว่า {GPAX_MIN_GRADUATE:.2f} ติดต่อกัน {probation_terms} ภาคเรียน "
                f"(เกินเกณฑ์วิทยาทัณฑ์ {MAX_PROBATION_TERMS} ภาคเรียน) เข้าเกณฑ์พ้นสภาพนักศึกษา"
            )
        else:
            level = 1
            alerts.append(
                f"GPAX {gpax:.2f} อยู่ในช่วงวิทยาทัณฑ์ ({GPAX_DISMISS:.2f} - 1.99) ต่ำกว่า {GPAX_MIN_GRADUATE:.2f} "
                f"ติดต่อกัน {probation_terms} ภาคเรียน (พ้นสภาพเมื่อเกิน {MAX_PROBATION_TERMS} ภาคเรียน)"
            )
    if academic_year == 4 and gpax < GPAX_MIN_GRADUATE:
        alerts.append(f"ชั้นปี 4 แต่ GPAX ต่ำกว่า {GPAX_MIN_GRADUATE:.2f} ยังไม่สามารถสำเร็จการศึกษาได้")
    return level, alerts


def ประเมินความเสี่ยง(features, academic_year, n_terms):
    """ทำนายด้วย AI แล้วปรับระดับความเสี่ยงขึ้นตามเกณฑ์ข้อบังคับ (ใช้ระดับที่สูงกว่า)"""
    input_df = pd.DataFrame([features])[list(model.feature_names_in_)]

    prob = float(model.predict_proba(input_df)[0][1])
    score_percent = round(prob * 100, 2)
    if score_percent >= 51.0:
        ai_level = 2
    elif score_percent >= 41.0:
        ai_level = 1
    else:
        ai_level = 0

    rule_level, alerts = ตรวจเกณฑ์ข้อบังคับ(features['gpax'], features['probation_terms'], academic_year, n_terms)
    if rule_level > ai_level:
        alerts.append("ระดับความเสี่ยงถูกปรับขึ้นตามเกณฑ์ข้อบังคับการพ้นสภาพ")
    risk_status = RISK_LEVELS[max(ai_level, rule_level)]

    shap_values = explainer(input_df)
    key_factors = []
    for name, val, shap_val in zip(input_df.columns, input_df.iloc[0], shap_values.values[0]):
        if hasattr(shap_val, "__len__"):
            shap_val = float(shap_val[1]) if len(shap_val) > 1 else float(shap_val[0])
        else:
            shap_val = float(shap_val)
        key_factors.append(
            ปัจจัยความเสี่ยงจากAI(
                feature_name=thai_names.get(name, name),
                current_value=float(val),
                risk_contribution=shap_val
            )
        )
    key_factors.sort(key=lambda x: abs(x.risk_contribution), reverse=True)

    return score_percent, risk_status, key_factors, alerts


def บันทึกผลการประเมิน(db, student_id, first_name, last_name, academic_year,
                      term_results, credits_earned, credits_planned):
    features = สร้างฟีเจอร์(term_results, credits_earned, credits_planned)
    db_student = db.query(Student).filter(Student.student_id == student_id).first()
    if not db_student:
        db_student = Student(
            student_id=student_id,
            first_name=first_name,
            last_name=last_name,
            highschool_gpa=features['gpax'],
            academic_year=academic_year
        )
        db.add(db_student)
    else:
        db_student.highschool_gpa = features['gpax']
        if academic_year is not None:
            db_student.academic_year = academic_year
    db.commit()

    score_percent, risk_status, key_factors, alerts = ประเมินความเสี่ยง(features, academic_year, len(term_results))

    db_prediction = RiskPrediction(
        student_id=student_id,
        risk_score=score_percent,
        risk_level=risk_status,
        shap_explanation={
            "ปัจจัยหลัก": [f.dict() for f in key_factors],
            "ข้อมูลที่ใช้ทำนาย": features,
            "ผลการเรียนรายภาค": term_results,
            "เกณฑ์ข้อบังคับ": alerts,
        }
    )
    db.add(db_prediction)
    db.commit()
    return score_percent, risk_status, key_factors, alerts


@router.post(
    "/predict-student",
    response_model=ผลการทำนายความเสี่ยง,
    summary="ทำนายความเสี่ยงและบันทึกข้อมูลนักศึกษา"
)
def ทำนายและบันทึกข้อมูลนักศึกษา(data: ข้อมูลนักศึกษาขาเข้า, db: Session = Depends(get_db)):
    if model is None or explainer is None:
        raise HTTPException(status_code=500, detail="ยังไม่ได้โหลดโมเดล AI ในระบบ")

    try:
        score_percent, risk_status, key_factors, alerts = บันทึกผลการประเมิน(
            db, data.student_id, data.first_name, data.last_name, data.academic_year,
            [t.model_dump() for t in data.term_results],
            data.credits_earned, data.credits_planned
        )

        return ผลการทำนายความเสี่ยง(
            student_id=data.student_id,
            student_name=f"{data.first_name} {data.last_name}",
            risk_score_percentage=score_percent,
            risk_status=risk_status,
            key_factors=key_factors,
            rule_alerts=alerts
        )
    except Exception as e:
        db.rollback()
        print("❌ เกิดข้อผิดพลาดขณะทำนายผล:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/students", summary="ดึงรายชื่อนักศึกษาทั้งหมด")
def ดึงรายชื่อนักศึกษา(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    results = []
    for s in students:
        latest_pred = db.query(RiskPrediction).filter(RiskPrediction.student_id == s.student_id).order_by(RiskPrediction.predicted_at.desc()).first()
        results.append({
            "student_id": s.student_id,
            "name": f"{s.first_name} {s.last_name}",
            "academic_year": s.academic_year,
            "risk_score": latest_pred.risk_score if latest_pred else 0,
            "risk_status": latest_pred.risk_level if latest_pred else "ยังไม่ได้ประเมิน"
        })
    return results

@router.delete("/delete-student/{student_id}", summary="ลบนักศึกษา")
def ลบนักศึกษา(student_id: str, db: Session = Depends(get_db)):
    db.query(RiskPrediction).filter(RiskPrediction.student_id == student_id).delete()
    db.query(Student).filter(Student.student_id == student_id).delete()
    db.commit()
    return {"message": "ลบข้อมูลสำเร็จ"}

@router.delete("/delete-all-students", summary="ลบข้อมูลนักศึกษาและประวัติการทำนายทั้งหมด")
def ลบข้อมูลนักศึกษาทั้งหมด(db: Session = Depends(get_db)):
    try:
        db.query(RiskPrediction).delete()
        db.query(Student).delete()
        db.commit()
        return {"message": "ลบข้อมูลนักศึกษาและประวัติทั้งหมดในระบบเรียบร้อยแล้ว"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export-excel", summary="ดาวน์โหลดรายชื่อนักศึกษาทั้งหมดเป็นไฟล์ Excel")
def ดาวน์โหลดไฟล์Excel(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    data_list = []

    for s in students:
        latest_pred = db.query(RiskPrediction).filter(RiskPrediction.student_id == s.student_id).order_by(RiskPrediction.predicted_at.desc()).first()
        explanation = (latest_pred.shap_explanation or {}) if latest_pred else {}
        data_list.append({
            "รหัสนักศึกษา": s.student_id,
            "ชื่อ-นามสกุล": f"{s.first_name} {s.last_name}",
            "ชั้นปี": s.academic_year if s.academic_year and 1 <= s.academic_year <= 4 else "-",
            "เกรดเฉลี่ย (GPAX)": s.highschool_gpa,
            "คะแนนความเสี่ยง (%)": latest_pred.risk_score if latest_pred else 0,
            "ระดับความเสี่ยง": latest_pred.risk_level if latest_pred else "ยังไม่ได้ประเมิน",
            "แจ้งเตือนตามข้อบังคับ": " | ".join(explanation.get("เกณฑ์ข้อบังคับ", [])) or "-",
            "วันที่ประเมินล่าสุด": str(latest_pred.predicted_at) if latest_pred else "-"
        })

    if not data_list:
        data_list = [{"รหัสนักศึกษา": "-", "ชื่อ-นามสกุล": "-", "ชั้นปี": "-", "เกรดเฉลี่ย (GPAX)": "-", "คะแนนความเสี่ยง (%)": "-", "ระดับความเสี่ยง": "-", "แจ้งเตือนตามข้อบังคับ": "-", "วันที่ประเมินล่าสุด": "-"}]

    df = pd.DataFrame(data_list)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='รายงานความเสี่ยงนักศึกษา')
    output.seek(0)

    headers = {'Content-Disposition': 'attachment; filename="student_risk_report.xlsx"'}
    return StreamingResponse(output, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers=headers)

@router.get("/import-template", summary="ดาวน์โหลดไฟล์ Excel ตัวอย่างสำหรับนำเข้าข้อมูล")
def ดาวน์โหลดเทมเพลตExcel():
    # ภาคเรียนที่ยังไม่ได้เรียนให้เว้นว่างไว้
    df = pd.DataFrame([
        {"student_id": "67123456", "first_name": "สมชาย", "last_name": "ใจดี", "academic_year": 2,
         "credits_earned": 45, "credits_planned": 54,
         "term1_gpax": 2.10, "term1_fw": 1, "term1_midterm": 60,
         "term2_gpax": 1.85, "term2_fw": 1, "term2_midterm": 52,
         "term3_gpax": 1.75, "term3_fw": 2, "term3_midterm": 48},
        {"student_id": "67123457", "first_name": "สมหญิง", "last_name": "ตั้งใจ", "academic_year": 1,
         "credits_earned": 18, "credits_planned": 18,
         "term1_gpax": 3.20, "term1_fw": 0, "term1_midterm": 78},
    ], columns=TEMPLATE_COLUMNS + TERM_COLUMNS)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='ข้อมูลนักศึกษา')
    output.seek(0)
    headers = {'Content-Disposition': 'attachment; filename="student_import_template.xlsx"'}
    return StreamingResponse(output, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers=headers)

def อ่านแถวเทมเพลต(row):
    """อ่านแถวจากไฟล์เทมเพลต คืนค่า None ถ้าข้อมูลไม่ครบหรือผิดช่วง"""
    try:
        academic_year = int(row['academic_year'])
        credits_earned = int(row['credits_earned'])
        credits_planned = int(row['credits_planned'])

        # อ่านผลรายภาคต่อกันตั้งแต่ภาค 1 จนเจอภาคที่เว้นว่าง
        term_results = []
        for t in range(1, MAX_TERMS + 1):
            gpax_val = row.get(f'term{t}_gpax')
            if gpax_val is None or pd.isna(gpax_val):
                break
            fw_val = row.get(f'term{t}_fw')
            fw_count = 0 if fw_val is None or pd.isna(fw_val) else int(fw_val)
            mid_val = row.get(f'term{t}_midterm')
            midterm_score = None if mid_val is None or pd.isna(mid_val) else float(mid_val)
            term_results.append({'gpax': float(gpax_val), 'fw_count': fw_count, 'midterm_score': midterm_score})
    except (TypeError, ValueError):
        return None
    if not term_results or not all(
        0 <= t['gpax'] <= 4 and t['fw_count'] >= 0
        and (t['midterm_score'] is None or 0 <= t['midterm_score'] <= 100)
        for t in term_results
    ):
        return None
    # ภาคล่าสุดต้องมีคะแนนเก็บ เพราะใช้ในการทำนาย
    if term_results[-1]['midterm_score'] is None:
        return None
    if not (1 <= academic_year <= 4 and credits_earned >= 0 and credits_planned > 0):
        return None
    student_id = str(row['student_id']).strip()
    if student_id.endswith('.0'):
        student_id = student_id[:-2]
    return (student_id, str(row['first_name']).strip(), str(row['last_name']).strip(), academic_year,
            term_results, credits_earned, credits_planned)

@router.post("/import-excel", summary="อัปโหลดไฟล์ Excel เพื่อนำเข้าและทำนายผลนักศึกษาหลายคน")
def นำเข้าไฟล์Excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if model is None or explainer is None:
        raise HTTPException(status_code=500, detail="ยังไม่ได้โหลดโมเดล AI ในระบบ")

    try:
        contents = file.file.read()
        df_header = pd.read_excel(io.BytesIO(contents), sheet_name=0)

        imported_count = 0
        skipped_count = 0

        # รูปแบบที่ 1: ไฟล์ตามเทมเพลต (มีครบทั้ง 5 ตัวแปรวิกฤต)
        if all(c in df_header.columns for c in TEMPLATE_COLUMNS + ['term1_gpax']):
            for _, row in df_header.iterrows():
                parsed = อ่านแถวเทมเพลต(row)
                if parsed is None or not parsed[0] or parsed[0] == 'nan':
                    skipped_count += 1
                    continue
                บันทึกผลการประเมิน(db, *parsed)
                imported_count += 1

            message = f"นำเข้าและประเมินผลสำเร็จ {imported_count} คน"
            if skipped_count:
                message += f" (ข้ามแถวที่ข้อมูลไม่ครบหรือผิดช่วง {skipped_count} แถว)"
            return {"message": message}

        # รูปแบบที่ 2: ใบคะแนนรายวิชาแบบเดิม (ลำดับ, ชื่อ, คะแนนเข้าเรียน, เต็ม, คะแนนสอบ, เต็ม)
        # ไม่มีข้อมูลหน่วยกิต วิชา F/W และวิทยาทัณฑ์ จึงใช้ค่าปกติแทน
        df_excel = pd.read_excel(io.BytesIO(contents), sheet_name=0, header=None)
        for idx in range(len(df_excel)):
            row = df_excel.iloc[idx]
            col_idx = row.iloc[0]
            full_name = str(row.iloc[1]).strip()

            if not full_name or full_name == 'nan' or 'ชื่อ' in full_name or 'เฉลี่ย' in full_name:
                continue

            if not str(col_idx).replace('.','',1).isdigit():
                continue

            student_id = f"STD_{int(float(col_idx))}"

            name_parts = full_name.split(' ', 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else "-"

            try:
                att_actual = float(row.iloc[2])
                att_max = float(row.iloc[3]) if float(row.iloc[3]) > 0 else 20.0
                exam_score = float(row.iloc[4])
                exam_max = float(row.iloc[5]) if float(row.iloc[5]) > 0 else 80.0
            except:
                att_actual, att_max, exam_score, exam_max = 18.0, 20.0, 70.0, 80.0

            gpax = round((exam_score / exam_max) * 4.0, 2)
            gpax = min(max(gpax, 0.0), 4.0)
            midterm_score = min(max((att_actual + exam_score) / (att_max + exam_max) * 100, 0.0), 100.0)

            บันทึกผลการประเมิน(db, student_id, first_name, last_name, None,
                              [{'gpax': gpax, 'fw_count': 0, 'midterm_score': midterm_score}], 1, 1)
            imported_count += 1

        return {"message": f"นำเข้าไฟล์คะแนนและประเมินผลสำเร็จทั้งหมด {imported_count} คน (ไฟล์รูปแบบเดิม: ไม่มีข้อมูลหน่วยกิต วิชา F/W และวิทยาทัณฑ์)"}
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
