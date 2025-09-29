from flask import Flask, render_template
from dummy_data import students, gamification_data, students_progress

app = Flask(__name__)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")

# ---------------- PARENT PORTAL ----------------
@app.route("/parent")
def parent_dashboard():
    student = students["student1"]

    completed = sum(1 for a in student["assignments"] if a["status"] == "completed")
    pending = sum(1 for a in student["assignments"] if a["status"] == "pending")

    analytics = []
    if student["attendance"] > 90:
        analytics.append("🎉 Excellent attendance! Keep it up.")
    else:
        analytics.append("⚠️ Attendance needs improvement.")

    avg_grade = sum(student["grades"].values()) / len(student["grades"])
    if avg_grade >= 85:
        analytics.append("📊 Strong academic performance overall.")
    else:
        analytics.append("📈 Focus on improving grades in some subjects.")

    if pending > completed:
        analytics.append("⏳ More pending work than completed, needs attention.")
    else:
        analytics.append("✅ Good progress on assignments.")

    return render_template(
        "parent.html",
        student=student,
        completed=completed,
        pending=pending,
        grades_labels=list(student["grades"].keys()),
        grades_values=list(student["grades"].values()),
        analytics=analytics
    )

# ---------------- GAMIFICATION ----------------
@app.route("/gamification")
def gamification_dashboard():
    student = gamification_data["student1"]
    leaderboard = gamification_data["leaderboard"]

    insights = []
    if student["xp_progress"] >= 70:
        insights.append("🔥 You're close to leveling up! Keep going.")
    if student["points"] >= 1000:
        insights.append("🏆 Impressive! You're among the top achievers.")
    if len(student["badges"]) >= 3:
        insights.append("🎖️ Amazing! You’ve unlocked multiple badges.")

    return render_template(
        "gamification.html",
        student=student,
        leaderboard=leaderboard,
        insights=insights
    )

# ---------------- PROGRESS TRACKING ----------------
@app.route("/progress")
def progress_dashboard():
    student = students_progress["student1"]

    analytics = []
    if student["course_completion"] >= 80:
        analytics.append("🎉 You are close to completing your course!")
    else:
        analytics.append("📈 Keep pushing forward, only a little more to complete the course!")

    avg_score = sum(student["scores"].values()) / len(student["scores"])
    if avg_score >= 85:
        analytics.append("📊 Excellent scores across subjects.")
    elif avg_score >= 70:
        analytics.append("👍 Good progress, but there is room for improvement.")
    else:
        analytics.append("⚠️ Scores are below expectations, focus more on weak areas.")

    if student["attendance"] >= 90:
        analytics.append("✅ Attendance is outstanding!")
    elif student["attendance"] >= 75:
        analytics.append("📅 Attendance is decent, aim for 90%+.")
    else:
        analytics.append("⚠️ Low attendance, attend more classes regularly.")

    if student["study_streak"] > 3:
        analytics.append(f"🔥 Amazing! {student['study_streak']} days continuous study streak.")
    else:
        analytics.append("📖 Try to maintain a consistent study streak.")

    weak_subjects = [sub for sub, score in student["scores"].items() if score < 70]
    if weak_subjects:
        analytics.append("🎯 Recommended: Spend more time on " + ", ".join(weak_subjects) + ".")

    return render_template(
        "progress.html",
        student=student,
        analytics=analytics,
        time_labels=list(student["time_spent"].keys()),
        time_values=list(student["time_spent"].values()),
        score_labels=list(student["scores"].keys()),
        score_values=list(student["scores"].values()),
        deadline_labels=list(student["deadlines"].keys()),
        deadline_values=list(student["deadlines"].values()),
        quiz_labels=list(student["quiz_attempts"].keys()),
        quiz_values=list(student["quiz_attempts"].values()),
        login_labels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        login_values=student["login_frequency"]
    )

if __name__ == "__main__":
    app.run(debug=True)
