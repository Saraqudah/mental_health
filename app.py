from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import traceback
import sqlite3
import random
from datetime import datetime
import os

app = Flask(__name__)

# تحميل البيانات وتدريب النماذج
try:
    df = pd.read_excel("data.xlsx")
    print("✅ تم تحميل البيانات بنجاح")

    feature_cols = ['Age', 'Gender', 'Mood Score (1-10)', 'Sleep Quality (1-10)',
                    'Physical Activity (hrs/week)', 'Stress Level (1-10)']
    
    target_cols = ['Diagnosis', 'AI-Detected Emotional State', 'Outcome', 'Medication', 'Therapy Type']

    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    models = {}

    X = df[feature_cols]

    for target in target_cols:
        y = df[target]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)

        models[target] = {
            'model': model,
            'encoder': le,
            'features': feature_cols
        }

    print("✅ تم تدريب النماذج بنجاح.")
except Exception as e:
    print("❌ حصل خطأ أثناء تجهيز البيانات:")
    traceback.print_exc()

# إنشاء الجدول إذا لم يكن موجودًا
def create_table_if_not_exists():
    try:
        conn = sqlite3.connect('data/mental_health.db', timeout=10)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS database (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, email TEXT, marital_status TEXT, location TEXT, education TEXT,
                age INTEGER, gender TEXT, mood_score INTEGER, sleep_quality INTEGER,
                physical_activity INTEGER, stress_level INTEGER,
                diagnosis TEXT, emotion TEXT, outcome TEXT, medication TEXT, therapy TEXT,
                treatment_start_date TEXT, treatment_duration_weeks INTEGER,
                adherence REAL, progress REAL
            )
        ''')
        conn.commit()
    except Exception as e:
        print("❌ خطأ أثناء إنشاء أو تحديث الجدول:")
        traceback.print_exc()
    finally:
        conn.close()

@app.route('/')
def home():
    return render_template('Page0.html')

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

        # المدخلات
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

        full_input = {
            'Age': age,
            'Gender': gender_encoded,
            'Mood Score (1-10)': mood_score,
            'Sleep Quality (1-10)': sleep_quality,
            'Physical Activity (hrs/week)': physical_activity,
            'Stress Level (1-10)': stress_level
        }

        results = {}
        for target in target_cols:
            model = models[target]['model']
            le = models[target]['encoder']
            features = models[target]['features']
            input_df = pd.DataFrame([{k: full_input[k] for k in features}])
            prediction_encoded = model.predict(input_df)[0]
            prediction = le.inverse_transform([prediction_encoded])[0]
            results[target] = prediction

        conn = sqlite3.connect('data/mental_health.db', timeout=10)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT mood_score, sleep_quality, stress_level, emotion, treatment_start_date, treatment_duration_weeks
            FROM database WHERE email = ? ORDER BY id ASC LIMIT 1
        """, (email,))
        row = cursor.fetchone()

        emotion_state = results['AI-Detected Emotional State']

        # رسائل المشاعر
        emotion_messages = {
            "Stressed": "ou're feeling Stressed —It's okay to feel stressed sometimes — it's part of being human. Take a deep breath and remind yourself that you're doing your best,and that’s enough. Things will calm down, and you are stronger than you think.🌿\nلا بأس أن تشعر بالتوتر أحيانًا، فهذا جزء من كونك إنسانًا. خذ نفسًا عميقًا، وذكّر نفسك أنك تبذل ما بوسعك، وهذا يكفي. الأمور ستهدأ، وأنت أقوى مما تعتقد.🌿",
            "Happy": "It's wonderful to feel happy! Embrace that joy with gratitude and let your light inspire those around you. Remember, beautiful moments are meant to be fully lived and cherished. Smile — you're spreading positive energy without even trying 😊☀\nجميل أن تشعر بالسعادة! احتضن هذا الشعور بكل امتنان، ودَع نورك يُلهم من حولك. تذكّر أن اللحظات الجميلة تستحق أن نعيشها بوعي وفرح كامل. ابتسم... فأنت تنشر طاقة إيجابية دون أن تدري 😊☀",
            "Anxious": "Anxiety can feel overwhelming, but it doesn’t define you or control you. Remember — thoughts aren’t facts, and this feeling will pass. Take it one moment at a time, and give yourself the peace you deserve. You are safe right now.🕊\nقلق شعور مزعج، لكنه لا يُعرّفك ولا يتحكم بك. تذكّر أن الأفكار ليست حقائق، وأن كل شيء يمر—even هذا القلق. خذ الأمور لحظة بلحظة، وامنح نفسك الطمأنينة التي تستحقها. أنت بأمان الآن. 🕊",
            "Depressed": "I know that feeling depressed can make everything feel heavy... but you're not alone. Your presence matters — even on the days it doesn't feel like it. It's okay to rest, and to give yourself time to heal. One small step is enough today. 🌧💙\nأعلم أن الشعور بالاكتئاب يمكن أن يجعل كل شيء يبدو ثقيلاً... لكنك لست وحدك. وجودك مهم، حتى في الأيام التي لا تشعر فيها بذلك. لا بأس أن تطلب الراحة، وأن تمنح نفسك وقتًا للتعافي. خطوة صغيرة واحدة كافية اليوم. 🌧💙",
            "Excited": "It's amazing that you're feeling excited! That spark means something truly matters to you. Enjoy the moment, and celebrate every step — no matter how small. You're on a path full of possibilities! ⚡🎉\nرائع أنك تشعر بالحماس! هذه الطاقة الجميلة هي علامة على شيء يهمك حقًا. استمتع بكل لحظة، واسمح لنفسك أن تحتفل بخطواتك مهما كانت صغيرة. أنت على طريق مليء بالإمكانيات! ⚡🎉",
            "Neutral": "Feeling calm and connected is a true gift. Savor this balance and be present with it. Like nature, you grow quietly and bloom in your own time. Let this peace guide you. 🍃🌿\nأن تشعر بالسلام الداخلي والانسجام مع نفسك هو نعمة حقيقية. استمتع بهذا الاتزان، وخذ لحظاتك بكل وعي. مثل الطبيعة، أنت تنمو بهدوء، وتزدهر دون استعجال. دَع هذا الهدوء يرشدك. 🍃🌿"
        }

        emotion_message = emotion_messages.get(emotion_state, "")
        is_new_user = row is None

        if is_new_user:
            treatment_start_date = datetime.today().strftime('%Y-%m-%d')
            diagnosis = results['Diagnosis']
            duration_ranges = {
                "Major Depressive Disorder": (10, 16),
                "Generalized Anxiety": (10, 18),
                "Bipolar Disorder": (12, 20),
                "Panic Disorder": (8, 14)
            }
            default_range = (8, 12)
            selected_range = duration_ranges.get(diagnosis, default_range)
            treatment_duration_weeks = random.randint(*selected_range)
            adherence = 0
            progress = None
            weeks_remaining = treatment_duration_weeks
            continue_message = ""
        else:
            old_mood, old_sleep, old_stress, previous_emotion, start_date_str, duration_weeks = row
            treatment_start_date = start_date_str
            treatment_duration_weeks = duration_weeks
            improved = 0
            if mood_score > old_mood: improved += 1
            if sleep_quality > old_sleep: improved += 1
            if stress_level < old_stress: improved += 1

            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            today = datetime.today()
            weeks_passed = max(0, (today - start_date).days // 7)

            if weeks_passed >= 1:
                raw_progress = (mood_score + sleep_quality + physical_activity - stress_level) / 3
                raw_progress = max(0, min(raw_progress, 10))

                if treatment_duration_weeks > 0:
                    weeks_ratio = min(weeks_passed / treatment_duration_weeks, 1.0)
                    progress = round(raw_progress * weeks_ratio)
                else:
                    progress = 0
            else:
                progress = 0

            improvement_ratio = improved / 3
            adherence = improvement_ratio * min(weeks_passed / treatment_duration_weeks, 1.0) * 100 if treatment_duration_weeks else 0
            weeks_remaining = max(0, treatment_duration_weeks - weeks_passed)
            continue_message = "استمر بالعلاج 💪"

        # حفظ في قاعدة البيانات
        cursor.execute('''
            INSERT INTO database (
                name, email, marital_status, location, education,
                age, gender, mood_score, sleep_quality, physical_activity, stress_level,
                diagnosis, emotion, outcome, medication, therapy,
                treatment_start_date, treatment_duration_weeks, adherence, progress
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                name, email, marital_status, location, education,
                age, gender, mood_score, sleep_quality, physical_activity, stress_level,
                results['Diagnosis'], emotion_state, results['Outcome'], results['Medication'], results['Therapy Type'],
                treatment_start_date, treatment_duration_weeks, adherence, progress
            ))

        conn.commit()

        if not is_new_user and weeks_passed >= 1:
            progress_score = round((progress / 10) * 10)
            progress_score = max(1, min(progress_score, 10))
            progress_level = f"{progress_score}/10 - {'📈 ممتاز' if progress_score >= 8 else '🔄 متوسط' if progress_score >= 5 else '⚠ منخفض'}"
            adherence_level = "✅ عالي" if adherence >= 80 else "🟡 متوسط" if adherence >= 50 else "❌ منخفض"
        elif not is_new_user:
            progress_level = "🛛 لم يمضِ أسبوع على بدء العلاج"
            adherence_level = "ℹ لا يمكن حساب الالتزام"
        else:
            progress_level = "لا يوجد سجل سابق لقياس التقدم 💼"
            adherence_level = "ℹ لم يتم احتساب الالتزام"

        show_duration = not (
            results['Diagnosis'] == 'No Disorder' or
            results['Medication'] == 'None' or
            results['Therapy Type'] == 'None'
        )

        return render_template("Page3.html",
                               diagnosis=results['Diagnosis'],
                               emotion=emotion_state,
                               outcome=results['Outcome'] if not is_new_user else None,
                               medication=results['Medication'],
                               therapy=results['Therapy Type'],
                               emotion_message=emotion_message,
                               progress=progress if not is_new_user else None,
                               adherence=round(adherence, 2) if not is_new_user else None,
                               treatment_start_date=treatment_start_date,
                               treatment_duration_weeks=treatment_duration_weeks,
                               is_new_user=is_new_user,
                               progress_level=progress_level if not is_new_user else None,
                               adherence_level=adherence_level if not is_new_user else None,
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0",port=port) 
    
