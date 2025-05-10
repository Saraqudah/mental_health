from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import traceback
import sqlite3
import random
from datetime import datetime

app = Flask(__name__)

# تحميل البيانات وتدريب النموذج
try:
    df = pd.read_excel("data.xlsx")
    print("✅ تم تحميل البيانات بنجاح")

    feature_cols = ['Age', 'Gender', 'Mood Score (1-10)', 'Sleep Quality (1-10)',
                    'Physical Activity (hrs/week)', 'Stress Level (1-10)']
    target_cols = ['Diagnosis', 'AI-Detected Emotional State', 'Outcome', 'Medication', 'Therapy Type']

    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    X = df[feature_cols]
    models = {}

    for target in target_cols:
        y = df[target]
        model = RandomForestClassifier()
        model.fit(X, y)
        models[target] = model

except Exception as e:
    print("❌ حصل خطأ أثناء تجهيز البيانات:")
    traceback.print_exc()

# إنشاء جدول قاعدة البيانات إذا لم يكن موجوداً، وإضافة الأعمدة الناقصة إن وُجدت
def create_table_if_not_exists():
    try:
        conn = sqlite3.connect('mental_health.db', timeout=10)
        cursor = conn.cursor()

        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS database (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, 
                email TEXT, 
                marital_status TEXT, 
                location TEXT, 
                education TEXT,
                age INTEGER, 
                gender TEXT, 
                mood_score INTEGER,
                sleep_quality INTEGER, 
                physical_activity INTEGER, 
                stress_level INTEGER,
                diagnosis TEXT, 
                emotion TEXT, 
                outcome TEXT, 
                medication TEXT, 
                therapy TEXT,
                treatment_start_date TEXT, 
                treatment_duration_weeks INTEGER
            )
        ''')

        # الحصول على أسماء الأعمدة الحالية
        cursor.execute("PRAGMA table_info(database)")
        existing_columns = [col[1] for col in cursor.fetchall()]

        # إضافة الأعمدة إذا كانت غير موجودة
        if 'adherence' not in existing_columns:
            cursor.execute("ALTER TABLE database ADD COLUMN adherence REAL")
        if 'progress' not in existing_columns:
            cursor.execute("ALTER TABLE database ADD COLUMN progress REAL")

        conn.commit()
    except Exception as e:
        print("❌ خطأ أثناء إنشاء أو تحديث الجدول:")
        traceback.print_exc()
    finally:
        conn.close()

@app.route('/')
def home():
    return render_template('Page1.html')

@app.route('/Page1')
def page1():
    return render_template('Page1.html')

@app.route('/save_personal_info', methods=['POST'])
def save_personal_info():
    return render_template('Page2.html',
                           name=request.form['name'],
                           email=request.form['email'],
                           marital_status=request.form['marital_status'],
                           location=request.form['location'],
                           education=request.form['education'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        create_table_if_not_exists()

        age = int(request.form['age'])
        gender = request.form['gender']
        mood_score = int(request.form['mood_score'])
        sleep_quality = int(request.form['sleep_quality'])
        physical_activity = int(request.form['physical_activity'])
        stress_level = int(request.form['stress_level'])

        name = request.form['name']
        email = request.form['email']
        marital_status = request.form['marital_status']
        location = request.form['location']
        education = request.form['education']

        gender_encoded = le_gender.transform([gender])[0]

        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender_encoded,
            'Mood Score (1-10)': mood_score,
            'Sleep Quality (1-10)': sleep_quality,
            'Physical Activity (hrs/week)': physical_activity,
            'Stress Level (1-10)': stress_level
        }])

        results = {}
        progress = None
        adherence = 0
        
        if 8 <= mood_score <= 10 or 8 <= sleep_quality <= 10 or 8 <= physical_activity <= 10 and 1 <= stress_level <= 2:
            results['AI-Detected Emotional State'] = 'Happy'
            results['Diagnosis'] = 'No Disorder'
            results['Outcome'] = 'Stable'
            results['Medication'] = 'None'
            results['Therapy Type'] = 'None' 
            
            outcome_display = results['Outcome']
        else:
            for target in target_cols:
                prediction = models[target].predict(input_df)[0]
                results[target] = prediction
            outcome_display = results.get('Outcome', "Unknown Outcome")

        duration_ranges = {
            "Major Depressive Disorder": (10, 16),
            "Generalized Anxiety": (10, 18),
            "Bipolar Disorder": (12, 20),
            "Panic Disorder": (8, 14)
        }

        default_range = (8, 12)
        selected_range = duration_ranges.get(results['Diagnosis'], default_range)
        treatment_duration_weeks = random.randint(*selected_range)

        conn = sqlite3.connect('mental_health.db', timeout=10)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT mood_score, sleep_quality, stress_level, emotion, treatment_start_date, treatment_duration_weeks
            FROM database WHERE email = ? ORDER BY id DESC LIMIT 1
        """, (email,))
        row = cursor.fetchone()

        emotion_state = results['AI-Detected Emotional State']
        emotion_messages = {
            "Happy": "😊 You're feeling happy — let that smile light up the world!\n😊 أنت تشعر بالسعادة — دع ابتسامتك تنير العالم!",
            "Depressed": "😞 It looks like you're feeling depressed. You're not alone — seek support when needed.\n😞 يبدو أنك تشعر بالاكتئاب. لست وحدك — اطلب الدعم عند الحاجة.",
            "Neutral": "😐 You're in a neutral emotional state — maintaining balance is valuable!\n😐 أنت في حالة عاطفية محايدة — الحفاظ على التوازن أمر مهم!"
        }
        emotion_message = emotion_messages.get(emotion_state, "")

        if row is None:
            treatment_start_date = datetime.today().strftime('%Y-%m-%d')
            outcome_display = "No Outcome, Because you are a new user"
            progress = None
            adherence = 0
            weeks_remaining = None
            continue_message = ""
        else:
            old_mood, old_sleep, old_stress, previous_emotion, start_date_str, duration_weeks = row
            treatment_start_date = start_date_str

            improved = 0
            if mood_score > old_mood: improved += 1
            if sleep_quality > old_sleep: improved += 1
            if stress_level < old_stress: improved += 1

            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                today = datetime.today()
                weeks_passed = (today - start_date).days // 7
                weeks_passed = max(1, weeks_passed)

                progress = round((mood_score + sleep_quality + physical_activity - stress_level) / 3)
                progress = max(1, min(progress, 10))

                total_indicators = 3
                improvement_ratio = improved / total_indicators
                adherence = improvement_ratio * min(weeks_passed / duration_weeks, 1.0) * 100

                weeks_remaining = max(0, duration_weeks - weeks_passed)
                continue_message = "استمر بالعلاج 💪"
            except:
                progress = 0
                adherence = 0
                weeks_remaining = None
                continue_message = ""

        cursor.execute('''
            INSERT INTO database (
                name, email, marital_status, location, education,
                age, gender, mood_score,
                sleep_quality, physical_activity, stress_level,
                diagnosis, emotion, outcome, medication, therapy,
                treatment_start_date, treatment_duration_weeks, adherence, progress
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                name, email, marital_status, location, education,
                age, gender, mood_score,
                sleep_quality, physical_activity, stress_level,
                results['Diagnosis'], emotion_state,
                outcome_display, results['Medication'], results['Therapy Type'],
                treatment_start_date, treatment_duration_weeks, adherence, progress
            ))

        conn.commit()

        is_new_user = row is None
        if not is_new_user:
            progress_score = round((progress / 100) * 10)
            progress_score = max(1, min(progress_score, 10))

            if progress_score >= 8:
                progress_level = f"{progress_score}/10 - 📈 ممتاز"
            elif progress_score >= 5:
                progress_level = f"{progress_score}/10 - 🔄 متوسط"
            else:
                progress_level = f"{progress_score}/10 - ⚠ منخفض"

            if adherence >= 80:
                adherence_level = "✅ عالي"
            elif adherence >= 50:
                adherence_level = "🟡 متوسط"
            else:
                adherence_level = "❌ منخفض"
        else:
            progress_level = "لا يوجد سجل سابق لقياس التقدم 💼"
            adherence_level = "لا يمكن حساب الالتزام لمستخدم جديد ℹ"

        show_duration = not (
            results['Diagnosis'] == 'No Disorder' or
            results['Medication'] == 'None' or
            results['Therapy Type'] == 'None'
        )

        return render_template("Page3.html",
                               diagnosis=results['Diagnosis'],
                               emotion=emotion_state,
                               outcome=outcome_display,
                               medication=results['Medication'],
                               therapy=results['Therapy Type'],
                               emotion_message=emotion_message,
                               progress=progress if not is_new_user else None,
                               adherence=round(adherence, 2) if not is_new_user else None,
                               treatment_start_date=treatment_start_date,
                               treatment_duration_weeks=treatment_duration_weeks,
                               is_new_user=is_new_user,
                               progress_level=progress_level,
                               adherence_level=adherence_level,
                               weeks_remaining=weeks_remaining if not is_new_user else None,
                               continue_message=continue_message if not is_new_user else "")

    except Exception as e:
        print("❌ خطأ أثناء التنبؤ أو الحفظ:")
        traceback.print_exc()
        return "حدث خطأ أثناء المعالجة. تحقق من المدخلات."
    finally:
        try:
            conn.close()
        except:
            pass

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', use_reloader=False)