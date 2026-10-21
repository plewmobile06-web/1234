from sqlalchemy import Column, String, Numeric, Integer, TIMESTAMP, ForeignKey, JSON
from sqlalchemy.sql import func
from app.core.database import Base

class Student(Base):
    __tablename__ = "students"

    student_id = Column(String(20), primary_key=True, index=True)
    first_name = Column(String(100))
    last_name = Column(String(100))
    advisor_id = Column(String(20))
    highschool_gpa = Column(Numeric(3, 2))
    admission_type = Column(String(50))
    academic_year = Column(Integer)
    created_at = Column(TIMESTAMP, server_default=func.now())

class RiskPrediction(Base):
    __tablename__ = "risk_predictions"

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(20), ForeignKey("students.student_id"))
    risk_score = Column(Numeric(5, 2))
    risk_level = Column(String(20))
    shap_explanation = Column(JSON)
    predicted_at = Column(TIMESTAMP, server_default=func.now())