# Premier League Standings & Winner Predictor (2019–2029)

An end-to-end machine learning web application built on **AIMLite (v2.1.2)** using Dixon-Coles Poisson Monte Carlo simulation. Predicts the complete 20-team Premier League table, point tallies, goal differences, and champions for seasons from 2019 up to 2029.

---

## 🚀 Running with Docker (Recommended for Hosting)

### 1. Build the Docker Image
```bash
docker build -t pl-predictor .
```

The Docker build follows this streamlined pipeline:
1. Base image `python:3.12-slim`
2. Installs `uv` and `aimlite` via pip
3. Copies `aimlite.json` and runs `aimlite install` to fetch project dependencies (`numpy`, `scikit-learn`)
4. Copies project source files, dataset, and minimal dark frontend
5. Runs `aimlite train` to fit Poisson MLE model weights
6. Hosts the app using `aimlite serve --host 0.0.0.0 --port 8000 --frontend frontend`

### 2. Run the Container
```bash
docker run -d -p 8000:8000 --name pl-predictor pl-predictor
```

Access the application in your browser at `http://localhost:8000`.

### 3. Or use Docker Compose
```bash
docker compose up -d
```

To stop:
```bash
docker compose down
```

---

## 💻 Running Locally without Docker

```bash
# 1. Install project dependencies
aimlite install

# 2. Train model weights
aimlite train

# 3. Launch server with custom minimalist frontend
aimlite serve --host 0.0.0.0 --port 8000 --frontend frontend
```

