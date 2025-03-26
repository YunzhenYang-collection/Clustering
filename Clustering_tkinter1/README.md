# Cluster GUI: KMeans & DBSCAN Visual Tool (with Step-by-Step DBSCAN Tuning)

This interactive Python GUI tool allows users to explore clustering with **KMeans** and **DBSCAN**, using popular datasets. It features real-time visualization, step-by-step control, and automatic DBSCAN parameter tuning.

---

## 🧠 Principle Summary

### KMeans
- Assigns points to the nearest cluster center
- Updates centers as the mean of assigned points
- Stops when assignments no longer change

### DBSCAN
- Groups points based on density
- `eps`: neighborhood radius
- `min_samples`: minimum points to form a dense region
- Detects outliers (noise points)

---

## 💡 Features

- Dataset selection: `Iris`, `Penguins`, `make_moons`, `make_circles`
- Algorithm choice: `KMeans`, `DBSCAN`
- Clustering control:
  - KMeans: Manual / Automatic animation
  - DBSCAN: Step-by-step parameter tuning
- Real-time visualization with `matplotlib` embedded in `Tkinter`

---

## ▶️ How to Use

1. Run the script:
```bash
python Clustering_tkinter1.py
```

2. Select a dataset (e.g. `make_moons`)
3. Choose algorithm (default: `DBSCAN`)
4. Set **mode** to `DBSCAN Parameter Tuning`
5. Click **Start Observation**, then click **Next Step** to:
   - Increment `eps` by 0.05
   - Increment `min_samples` by 1
   - Re-cluster and display results (clusters, noise, parameters)

---

## 🔍 Code Highlights

### DBSCAN Step Control
```python
def step_once():
    global eps_val, min_samples_val, assignments
    db = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    assignments = db.fit_predict(X)
    update_plot(assignments)
    status_label.config(text=f"eps={eps_val:.2f}, min_samples={min_samples_val}")
    eps_val += 0.05
    min_samples_val += 1
```

### Canvas Update
```python
def update_plot(assignments):
    ax.clear()
    ...
    ax.scatter(...)  # colored clusters and noise
    canvas.draw()
```

---

## 📦 Requirements

- Python 3.8+
- matplotlib
- pandas
- seaborn
- scikit-learn
- tkinter

---

## 📜 License

 Apache2.0 License.
