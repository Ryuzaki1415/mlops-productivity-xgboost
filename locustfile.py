from locust import HttpUser, task, between
import time


import random

def generate_payload():
    return {
        "Daily_Phone_Hours": random.uniform(4, 10),
        "Social_Media_Hours": random.uniform(1, 5),
        "Sleep_Hours": random.uniform(5, 8),
        "Stress_Level": random.uniform(3, 9),
        "App_Usage_Count": random.randint(5, 30),
        "Caffeine_Intake_Cups": random.uniform(0, 4),
        "Weekend_Screen_Time_Hours": random.uniform(4, 12),
        "Gender": random.choice(["Male", "Female"]),
        "Occupation": random.choice(["Engineer", "Doctor", "Student"]),
        "Device_Type": random.choice(["Android", "iOS"])
    }

class PredictionUser(HttpUser):

    wait_time = between(1, 3)

    @task
    def predict_flow(self):

        # 1️⃣ Submit task
        response = self.client.post("/predict", json=generate_payload())
        if response.status_code != 200:
            return

        data = response.json()

        # If cached
        if "score" in data:
            return

        task_id = data.get("task_id")
        if not task_id:
            return

        # 2️⃣ Poll until complete
        while True:
            result = self.client.get(f"/result/{task_id}")

            if result.status_code == 202:
                time.sleep(1)
                continue

            break