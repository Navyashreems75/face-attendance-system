from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'

    id        = db.Column(db.Integer, primary_key=True)
    username  = db.Column(db.String(80), unique=True, nullable=False)
    password  = db.Column(db.String(200), nullable=False)
    role      = db.Column(db.String(20), nullable=False)  # admin / teacher / student
    full_name = db.Column(db.String(120), nullable=False)

class Subject(db.Model):
    __tablename__ = 'subjects'

    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(120), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    teacher    = db.relationship('User', foreign_keys=[teacher_id])

class SubjectAttendance(db.Model):
    __tablename__ = 'subject_attendance'

    id         = db.Column(db.Integer, primary_key=True)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date       = db.Column(db.String(10), nullable=False)  # YYYY-MM-DD
    time       = db.Column(db.String(8), nullable=True)