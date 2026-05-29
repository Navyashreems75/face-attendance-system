from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, make_response, send_file, abort
from models import db, User, Subject, SubjectAttendance
from functools import wraps
from datetime import date, timedelta, datetime
from werkzeug.security import generate_password_hash, check_password_hash
import os, base64, requests, csv, io, secrets

app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# ── Sub-Group A API ────────────────────────────────────────
AI_API_BASE = "http://localhost:5001"

# ── Dataset directory for student photos ──────────────────
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

# ── Decorators ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            if session.get('role') not in roles:
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated
    return decorator

# ── Helpers ────────────────────────────────────────────────
def name_to_csv_key(full_name, user_id=None):
    return full_name.replace(' ', '_')

def csv_key_matches(csv_row_name, name_prefix):
    return csv_row_name == name_prefix or csv_row_name.startswith(name_prefix + '_')

def get_all_csv_dates(days=60):
    today = date.today()
    dates = []
    for i in range(days):
        d = today - timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')
        if os.path.exists(f"attendance_{d_str}.csv"):
            dates.append(d_str)
    return sorted(dates)

def get_attendance_records(csv_key, days=60):
    seen_dates = set()
    records = []
    today = date.today()
    for i in range(days):
        d = today - timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')
        path = f"attendance_{d_str}.csv"
        if not os.path.exists(path):
            continue
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                if csv_key_matches(row['Name'], csv_key):
                    row_date = row.get('Date', d_str)
                    if row_date not in seen_dates:
                        seen_dates.add(row_date)
                        records.append({'date': row_date, 'time': row.get('Time', '')})
    return records

def get_working_days(days=60):
    today = date.today()
    count = 0
    for i in range(days):
        d = today - timedelta(days=i)
        if d > today:
            continue
        if os.path.exists(f"attendance_{d.strftime('%Y-%m-%d')}.csv"):
            count += 1
    return count

def compute_streaks(records):
    if not records:
        return 0, 0
    today = date.today()
    present_dates = sorted({date.fromisoformat(r['date']) for r in records}, reverse=True)

    streak, check = 0, today
    for d in present_dates:
        if d == check:
            streak += 1
            check -= timedelta(days=1)
            while check.weekday() >= 5:
                check -= timedelta(days=1)
        else:
            break

    sorted_asc = sorted(present_dates)
    best, run = (1, 1) if sorted_asc else (0, 0)
    for i in range(1, len(sorted_asc)):
        diff = (sorted_asc[i] - sorted_asc[i - 1]).days
        if diff == 1:
            run += 1
            best = max(best, run)
        elif diff > 1:
            run = 1
    best = max(best, run) if sorted_asc else 0

    return streak, best

def build_student_summary(students, all_dates, filter_date=None):
    date_list = [filter_date] if filter_date else all_dates

    csv_data = {}
    for d_str in date_list:
        path = f"attendance_{d_str}.csv"
        if not os.path.exists(path):
            csv_data[d_str] = {}
            continue
        present_map = {}
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                present_map[row['Name']] = row.get('Time', '')
        csv_data[d_str] = present_map

    today_str = date.today().strftime('%Y-%m-%d')
    summary   = []

    for student in students:
        csv_key = name_to_csv_key(student.full_name, student.id)
        records = {}

        for d_str in date_list:
            day_map = csv_data.get(d_str, {})
            matched_key = next((k for k in day_map if csv_key_matches(k, csv_key)), None)
            if matched_key is not None:
                records[d_str] = {'status': 'present', 'time': day_map[matched_key]}
            else:
                records[d_str] = {'status': 'absent',  'time': ''}

        present_count = sum(1 for v in records.values() if v['status'] == 'present')
        total         = len(date_list)
        present_count = min(present_count, total)
        absent_count  = max(0, total - present_count)
        pct           = round((present_count / total * 100) if total > 0 else 0, 1)
        pct           = min(pct, 100.0)
        today_status  = records.get(today_str, {}).get('status', '—') if today_str in date_list else '—'
        last_seen     = max((d for d, v in records.items() if v['status'] == 'present'), default=None)

        summary.append({
            'id':           student.id,
            'full_name':    student.full_name,
            'username':     student.username,
            'photo_url':    f"/student/photo/{student.id}",   # ← NEW
            'present':      present_count,
            'absent':       absent_count,
            'total':        total,
            'total_days':   total,
            'percentage':   pct,
            'today_status': today_status,
            'today':        today_status == 'present',
            'last_seen':    last_seen,
            'records':      records,
        })

    return summary

# ── Auth Routes ────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username  = request.form.get('username', '').strip()
        password  = request.form.get('password', '').strip()
        role      = request.form.get('role', '').strip()
        full_name = request.form.get('full_name', '').strip()

        if not all([username, password, role, full_name]):
            error = "All fields are required."
        elif role not in ('student', 'teacher', 'admin'):
            error = "Invalid role selected."
        elif User.query.filter_by(username=username).first():
            error = "Username already taken."
        else:
            hashed = generate_password_hash(password)
            db.session.add(User(
                username=username,
                password=hashed,
                role=role,
                full_name=full_name
            ))
            db.session.commit()
            flash("Account created — please log in.", "success")
            return redirect(url_for('login'))

    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id']  = user.id
            session['username'] = user.username
            session['role']     = user.role
            return redirect(url_for('dashboard'))

        error = "Invalid username or password."

    return render_template('login.html', error=error)

@app.route('/dashboard')
@login_required
def dashboard():
    role = session.get('role')
    if role == 'student':
        return redirect(url_for('student_dashboard'))
    elif role == 'teacher':
        return redirect(url_for('teacher_dashboard'))
    elif role == 'admin':
        return redirect(url_for('admin_panel'))
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    response = make_response(redirect(url_for('login')))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

# ── Student Photo ──────────────────────────────────────────
@app.route('/student/photo/<int:student_id>')
@login_required
def student_photo(student_id):
    """Serve a student's photo from their dataset folder."""
    student = db.session.get(User, student_id)
    if not student:
        abort(404)

    full_name = student.full_name  # e.g. "Navyashree M S"

    # Find the matching folder — could be "Name" or "Name_rollno"
    matched_folder = None
    if os.path.isdir(DATASET_DIR):
        for folder_name in sorted(os.listdir(DATASET_DIR)):
            if folder_name == full_name or folder_name.startswith(full_name + '_'):
                matched_folder = os.path.join(DATASET_DIR, folder_name)
                break

    if not matched_folder or not os.path.isdir(matched_folder):
        abort(404)

    # Return the first valid image in the folder
    for fname in sorted(os.listdir(matched_folder)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            return send_file(
                os.path.join(matched_folder, fname),
                mimetype='image/jpeg'
            )

    abort(404)

# ── Enroll / Kiosk ────────────────────────────────────────
@app.route('/enroll')
@login_required
def enroll():
    return render_template('enroll.html')

@app.route('/api/load-dataset', methods=['POST'])
@login_required
def load_dataset():
    data        = request.get_json()
    folder_name = data['folder']
    folder_path = os.path.join('dataset', folder_name)

    if not os.path.exists(folder_path):
        return jsonify({"error": "Folder not found"}), 404

    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
                images.append(f"data:image/jpeg;base64,{b64}")

    return jsonify({"images": images, "count": len(images)})

@app.route('/api/enroll', methods=['POST'])
@login_required
def api_enroll():
    data = request.get_json()
    try:
        resp = requests.post(
            f"{AI_API_BASE}/enroll",
            json={
                "name"  : data['name'],
                "roll"  : data.get('roll', ''),
                "images": data['images']
            },
            timeout=60
        )
        result = resp.json()
        if 'redirect' not in result:
            result['redirect'] = '/dashboard'
        return jsonify(result)
    except requests.exceptions.ConnectionError:
        return jsonify({"status": "error", "error": "AI service not running. Start api.py first."}), 503
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/kiosk')
@login_required
def kiosk():
    return render_template('kiosk.html')

@app.route('/api/recognize', methods=['POST'])
@login_required
def api_recognize():
    data       = request.get_json()
    session_id = data.get('session_id', f"user_{session.get('user_id', 'anon')}")

    try:
        resp   = requests.post(
            f"{AI_API_BASE}/recognize",
            json={"image": data['image'], "session_id": session_id},
            timeout=10
        )
        result = resp.json()
        return jsonify(result)

    except requests.exceptions.ConnectionError:
        return jsonify({"status": "error",
                        "error": "AI service not running. Start api.py first."}), 503
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

# ── Student Dashboard ──────────────────────────────────────
@app.route('/student/dashboard')
@role_required('student')
def student_dashboard():
    user         = db.session.get(User, session['user_id'])
    csv_key      = name_to_csv_key(user.full_name, user.id)
    raw_records  = get_attendance_records(csv_key, days=60)
    working_days = get_working_days(60)
    present_days = len(raw_records)
    present_days = min(present_days, working_days)
    absent_days  = max(0, working_days - present_days)
    percentage   = round((present_days / working_days * 100) if working_days > 0 else 0, 1)
    percentage   = min(percentage, 100.0)
    today_str    = date.today().strftime('%Y-%m-%d')
    today_bool   = any(r['date'] == today_str for r in raw_records)
    last_seen    = raw_records[0]['date'] if raw_records else None
    streak, best_streak = compute_streaks(raw_records)

    student = {
        'full_name':   user.full_name,
        'present':     present_days,
        'absent':      absent_days,
        'total_days':  working_days,
        'percentage':  percentage,
        'last_seen':   last_seen,
        'today':       today_bool,
        'streak':      streak,
        'best_streak': best_streak,
    }

    records = []
    present_date_set = {r['date'] for r in raw_records}
    today_obj = date.today()
    for i in range(60):
        d     = today_obj - timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')
        if not os.path.exists(f"attendance_{d_str}.csv"):
            continue
        if d_str in present_date_set:
            time_val = next((r['time'] for r in raw_records if r['date'] == d_str), None)
            records.append({'date': d_str, 'time': time_val, 'status': 'present'})
        else:
            records.append({'date': d_str, 'time': None, 'status': 'absent'})

    colors        = ['#EEF2FF', '#F0FDF4', '#FFF7ED', '#FDF4FF', '#F0F9FF', '#FFFBEB']
    border_colors = ['#6366F1', '#22C55E', '#F97316', '#A855F7', '#0EA5E9', '#EAB308']
    subject_stats = []
    for i, subj in enumerate(Subject.query.all()):
        subj_records   = SubjectAttendance.query.filter_by(subject_id=subj.id, student_id=user.id).all()
        s_present      = len(subj_records)
        all_subj_dates = db.session.query(SubjectAttendance.date).filter_by(
            subject_id=subj.id).distinct().count()
        s_absent  = max(0, all_subj_dates - s_present)
        s_pct     = round((s_present / all_subj_dates * 100) if all_subj_dates > 0 else 0, 1)
        today_present = SubjectAttendance.query.filter_by(
            subject_id=subj.id, student_id=user.id, date=today_str).first() is not None
        subject_stats.append({
            'name': subj.name, 'teacher': subj.teacher.full_name,
            'present': s_present, 'absent': s_absent,
            'total': all_subj_dates, 'percentage': s_pct,
            'today': 'Present' if today_present else ('Absent' if all_subj_dates > 0 else '—'),
            'color': colors[i % len(colors)],
            'border': border_colors[i % len(border_colors)]
        })

    return render_template('student_dashboard.html',
        user=user,
        student=student,
        records=records,
        subject_stats=subject_stats)

# ── Student Live Poll API ─────────────────────────────────
@app.route('/api/student/attendance')
@role_required('student')
def student_attendance_api():
    user         = db.session.get(User, session['user_id'])
    csv_key      = name_to_csv_key(user.full_name, user.id)
    raw_records  = get_attendance_records(csv_key, days=60)
    working_days = get_working_days(60)
    present_days = len(raw_records)
    present_days = min(present_days, working_days)
    absent_days  = max(0, working_days - present_days)
    percentage   = round((present_days / working_days * 100) if working_days > 0 else 0, 1)
    percentage   = min(percentage, 100.0)
    today_str    = date.today().strftime('%Y-%m-%d')
    today_bool   = any(r['date'] == today_str for r in raw_records)
    last_seen    = raw_records[0]['date'] if raw_records else None
    streak, best_streak = compute_streaks(raw_records)

    records = []
    present_date_set = {r['date'] for r in raw_records}
    today_obj = date.today()
    for i in range(60):
        d     = today_obj - timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')
        if not os.path.exists(f"attendance_{d_str}.csv"):
            continue
        if d_str in present_date_set:
            time_val = next((r['time'] for r in raw_records if r['date'] == d_str), None)
            records.append({'date': d_str, 'time': time_val, 'status': 'present'})
        else:
            records.append({'date': d_str, 'time': None, 'status': 'absent'})

    return jsonify({
        'student': {
            'full_name':   user.full_name,
            'present':     present_days,
            'absent':      absent_days,
            'total_days':  working_days,
            'percentage':  percentage,
            'last_seen':   last_seen,
            'today':       today_bool,
            'streak':      streak,
            'best_streak': best_streak,
        },
        'records': records
    })

# ── Teacher Dashboard ──────────────────────────────────────
@app.route('/teacher/dashboard')
@role_required('teacher', 'admin')
def teacher_dashboard():
    user        = db.session.get(User, session['user_id'])
    filter_date = request.args.get('date', None)
    all_dates   = get_all_csv_dates(days=60)
    students    = User.query.filter_by(role='student').order_by(User.full_name).all()

    if filter_date and filter_date not in all_dates:
        filter_date = None

    summary = build_student_summary(students, all_dates, filter_date=filter_date)

    daily_rows = []
    for d_str in reversed(all_dates):
        present_on_day = sum(
            1 for s in summary
            if s['records'].get(d_str, {}).get('status') == 'present'
        )
        daily_rows.append({
            'date':    d_str,
            'present': present_on_day,
            'absent':  len(students) - present_on_day,
            'total':   len(students),
        })

    return render_template('teacher_dashboard.html',
        user=user,
        summary=summary,
        daily_rows=daily_rows,
        all_dates=list(reversed(all_dates)),
        filter_date=filter_date)

# ── Teacher Live Poll API ─────────────────────────────────
@app.route('/api/teacher/summary')
@role_required('teacher', 'admin')
def teacher_summary_api():
    all_dates = get_all_csv_dates(days=60)
    students  = User.query.filter_by(role='student').order_by(User.full_name).all()
    summary   = build_student_summary(students, all_dates)

    today_str     = date.today().strftime('%Y-%m-%d')
    present_today = sum(1 for s in summary if s['today_status'] == 'present')
    at_risk       = sum(1 for s in summary if s['percentage'] < 75)
    avg_pct       = round(sum(s['percentage'] for s in summary) / len(summary), 1) if summary else 0

    return jsonify({
        'summary':       summary,
        'present_today': present_today,
        'at_risk':       at_risk,
        'avg_pct':       avg_pct,
        'total':         len(summary),
        'today':         today_str,
    })

@app.route('/teacher/export')
@role_required('teacher', 'admin')
def teacher_export():
    all_dates = get_all_csv_dates(days=60)
    students  = User.query.filter_by(role='student').order_by(User.full_name).all()
    summary   = build_student_summary(students, all_dates)

    output = io.StringIO()
    writer = csv.writer(output)

    header = ['Student Name'] + all_dates + ['Total Present', 'Total Absent', 'Percentage']
    writer.writerow(header)

    for s in summary:
        row = [s['full_name']]
        for d in all_dates:
            row.append('P' if s['records'].get(d, {}).get('status') == 'present' else 'A')
        row += [s['present'], s['absent'], f"{s['percentage']}%"]
        writer.writerow(row)

    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = (
        f'attachment; filename=attendance_export_{date.today()}.csv'
    )
    response.headers['Content-type'] = 'text/csv'
    return response

# ── Admin Panel ────────────────────────────────────────────
@app.route('/admin')
@role_required('admin')
def admin_panel():
    user      = db.session.get(User, session['user_id'])
    all_users = User.query.order_by(User.role, User.full_name).all()
    all_dates = get_all_csv_dates(days=60)
    students  = [u for u in all_users if u.role == 'student']
    summary   = build_student_summary(students, all_dates)

    total_students = len(students)
    today_str      = date.today().strftime('%Y-%m-%d')
    today_present  = sum(
        1 for s in summary
        if s['records'].get(today_str, {}).get('status') == 'present'
    ) if today_str in all_dates else 0
    below_75 = sum(1 for s in summary if s['percentage'] < 75)
    avg_pct  = round(
        sum(s['percentage'] for s in summary) / total_students, 1
    ) if total_students > 0 else 0.0

    stats = {
        'total_students': total_students,
        'today_present':  today_present,
        'today_absent':   total_students - today_present,
        'below_75':       below_75,
        'avg_percentage': avg_pct,
        'total_teachers': sum(1 for u in all_users if u.role == 'teacher'),
        'total_admins':   sum(1 for u in all_users if u.role == 'admin'),
        'working_days':   len(all_dates),
    }

    all_users_json = [
        {
            'id':        u.id,
            'username':  u.username,
            'full_name': u.full_name,
            'role':      u.role,
        }
        for u in all_users
    ]

    return render_template('admin_panel.html',
        user=user,
        summary=summary,
        all_users=all_users_json,
        stats=stats)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@role_required('admin')
def admin_delete_user(user_id):
    if user_id == session['user_id']:
        return jsonify({"status": "error", "error": "Cannot delete yourself."}), 400
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"status": "error", "error": "User not found."}), 404
    db.session.delete(user)
    db.session.commit()
    return jsonify({"status": "ok", "message": f"{user.full_name} deleted."})

@app.route('/admin/change_role/<int:user_id>', methods=['POST'])
@role_required('admin')
def admin_change_role(user_id):
    if user_id == session['user_id']:
        return jsonify({"status": "error", "error": "Cannot change your own role."}), 400
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"status": "error", "error": "User not found."}), 404
    new_role = request.get_json().get('role', '').strip()
    if new_role not in ('student', 'teacher', 'admin'):
        return jsonify({"status": "error", "error": "Invalid role."}), 400
    user.role = new_role
    db.session.commit()
    return jsonify({"status": "ok", "message": f"{user.full_name} is now {new_role}."})

@app.route('/admin/reset_password/<int:user_id>', methods=['POST'])
@role_required('admin')
def admin_reset_password(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"status": "error", "error": "User not found."}), 404
    new_password = request.get_json().get('password', '').strip()
    if len(new_password) < 4:
        return jsonify({"status": "error", "error": "Password must be at least 4 characters."}), 400
    user.password = generate_password_hash(new_password)
    db.session.commit()
    return jsonify({"status": "ok", "message": f"Password reset for {user.full_name}."})

# ── Manual mark ────────────────────────────────────────────
@app.route('/admin/manual_mark', methods=['POST'])
@app.route('/teacher/manual_mark', methods=['POST'])
@role_required('admin', 'teacher')
def admin_manual_mark():
    data      = request.get_json()
    full_name = data.get('student_name', '').strip()
    mark_date = data.get('date', '').strip()
    status    = data.get('status', '').strip()

    if not full_name or not mark_date or status not in ('present', 'absent'):
        return jsonify({"status": "error",
                        "error": "student_name, date, and status are required."}), 400

    try:
        datetime.strptime(mark_date, '%Y-%m-%d')
    except ValueError:
        return jsonify({"status": "error",
                        "error": "Invalid date format. Use YYYY-MM-DD."}), 400

    csv_key  = name_to_csv_key(full_name)
    csv_path = f"attendance_{mark_date}.csv"

    rows       = []
    fieldnames = ['Name', 'Date', 'Time']
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            reader     = csv.DictReader(f)
            fieldnames = reader.fieldnames or fieldnames
            rows       = [r for r in reader]

    already_present = any(csv_key_matches(r['Name'], csv_key) for r in rows)

    if status == 'present' and not already_present:
        rows.append({
            'Name': csv_key,
            'Date': mark_date,
            'Time': datetime.now().strftime('%H:%M:%S'),
        })
    elif status == 'absent' and already_present:
        rows = [r for r in rows if not csv_key_matches(r['Name'], csv_key)]
    else:
        msg = 'Already marked present.' if status == 'present' else 'Already marked absent.'
        return jsonify({"status": "ok", "message": msg})

    if status == 'absent' and not already_present:
        return jsonify({"status": "ok", "message": f"{full_name} was already absent for {mark_date}."})

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return jsonify({
        "status":  "ok",
        "message": f"{full_name} marked {status} for {mark_date}."
    })

# ── Run ────────────────────────────────────────────────────
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database tables created!")
    app.run(debug=False, port=5000)