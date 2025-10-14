# 🌬️ Lung Cancer Risk Prediction

Predict the risk of lung cancer using an interactive web application with a **FastAPI backend** and **Next.js frontend**.

---

## 🚀 Features

* 🔬 **Lung Cancer Risk Model** – Powered by FastAPI.
* 💻 **Interactive Frontend** – Next.js interface to input patient data and view predictions.
* 🐳 **Docker Support** – Easily run the app locally.
* 📊 **Input Features:**

  * `age`
  * `gender`
  * `pack_years` (smoking history)
  * `radon_exposure`
  * `asbestos_exposure`
  * `secondhand_smoke_exposure`
  * `copd_diagnosis`
  * `alcohol_consumption`
  * `family_history`

---

## 🛠️ Local Development

Run both backend and frontend locally using Docker Compose:

```bash
docker-compose up --build
```

* Backend: [http://localhost:8000](http://localhost:8000)
* Frontend: [http://localhost:3000](http://localhost:3000)

> ⚠️ Note: The Dockerfiles accept an `APP_DIR` argument for flexible build contexts. Docker Compose sets this automatically to the repository root.

---

## 💡 How to Use

1. Open the frontend in your browser: [http://localhost:3000](http://localhost:3000)
2. Fill in the patient data for the features listed above.
3. Click **Predict** to see the lung cancer risk.
4. Get instant feedback with clear risk indications.

---


