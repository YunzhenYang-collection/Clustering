import tkinter as tk
from tkinter import ttk
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

plt.style.use('default')  # 確保使用彩色樣式

# 全域變數
df = None
df_scaled = None
X = None
assignments = None
centers = None
iteration = 0
k = 3
canvas = None
ax = None
animation_running = False
feature_names = []

def update_feature_options(event=None):
    global df, df_scaled, feature_names

    dataset = dataset_var.get()
    if dataset == 'iris':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
    else:
        penguins = sns.load_dataset("penguins").dropna()
        df = penguins.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    feature_names = list(df_scaled.columns)
    feature1_menu['values'] = feature_names
    feature2_menu['values'] = feature_names

    feature1_var.set(feature_names[0])
    feature2_var.set(feature_names[1])

    update_preview()

def update_preview(event=None):
    global X, ax

    if not feature1_var.get() or not feature2_var.get():
        return

    f1, f2 = feature1_var.get(), feature2_var.get()
    X = df_scaled[[f1, f2]].values

    ax.clear()
    df_scaled.plot.scatter(x=f1, y=f2, ax=ax, color='blue', alpha=0.6, title="原始資料（未分群）")
    canvas.draw()

def reset_plot():
    global canvas, ax
    fig, ax = plt.subplots(figsize=(5, 5))
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack()
    canvas.draw()

def start_clustering():
    global X, centers, assignments, iteration, animation_running, k

    if animation_running:
        return

    k = int(k_var.get())
    f1, f2 = feature1_var.get(), feature2_var.get()
    X = df_scaled[[f1, f2]].values

    random_indices = random.sample(range(len(X)), k)
    centers = X[random_indices]
    assignments = np.zeros(len(X))
    iteration = 0
    animation_running = True
    animate_step()

def animate_step():
    global X, centers, assignments, iteration, animation_running, k

    new_assignments = np.array([
        np.argmin([np.linalg.norm(x - c) for c in centers])
        for x in X
    ])

    new_centers = np.array([
        X[new_assignments == i].mean(axis=0) if np.any(new_assignments == i) else centers[i]
        for i in range(k)
    ])

    update_plot(new_assignments, new_centers)

    changed = not np.array_equal(new_assignments, assignments)
    assignments[:] = new_assignments
    centers[:] = new_centers
    iteration += 1

    if changed:
        root.after(800, animate_step)
    else:
        animation_running = False
        print("✅ 分群已收斂")

def update_plot(assignments, centers):
    ax.clear()

    f1, f2 = feature1_var.get(), feature2_var.get()
    df_plot = df_scaled[[f1, f2]].copy()
    df_plot['cluster'] = assignments

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(k):
        cluster_df = df_plot[df_plot['cluster'] == i]
        cluster_df.plot.scatter(x=f1, y=f2, ax=ax, color=colors[i % len(colors)], label=f'群 {i}')
        center_x, center_y = centers[i]
        ax.scatter(center_x, center_y, s=200, color=colors[i % len(colors)], marker='X', edgecolors='black', linewidths=2)

    ax.set_title(f"KMeans Iteration {iteration}")
    canvas.draw()

# --- Tkinter UI 設定 ---
root = tk.Tk()
root.title("KMeans 分群動畫（彩色 + 可選特徵與群數）")

dataset_var = tk.StringVar(value='iris')
feature1_var = tk.StringVar()
feature2_var = tk.StringVar()
k_var = tk.StringVar(value='3')

frame = ttk.Frame(root)
frame.pack(padx=10, pady=10)

ttk.Label(frame, text="資料集：").grid(row=0, column=0)
dataset_menu = ttk.Combobox(frame, textvariable=dataset_var, values=["iris", "penguins"], state='readonly', width=10)
dataset_menu.grid(row=0, column=1)
dataset_menu.bind("<<ComboboxSelected>>", update_feature_options)

ttk.Label(frame, text="Feature 1：").grid(row=1, column=0)
feature1_menu = ttk.Combobox(frame, textvariable=feature1_var, state='readonly', width=15)
feature1_menu.grid(row=1, column=1)
feature1_menu.bind("<<ComboboxSelected>>", update_preview)

ttk.Label(frame, text="Feature 2：").grid(row=2, column=0)
feature2_menu = ttk.Combobox(frame, textvariable=feature2_var, state='readonly', width=15)
feature2_menu.grid(row=2, column=1)
feature2_menu.bind("<<ComboboxSelected>>", update_preview)

ttk.Label(frame, text="分群數 (k)：").grid(row=3, column=0)
ttk.Combobox(frame, textvariable=k_var, values=["2", "3", "4", "5"], state='readonly', width=5).grid(row=3, column=1)

ttk.Button(frame, text="開始分群（動畫）", command=start_clustering).grid(row=4, column=0, columnspan=2, pady=5)

canvas_frame = ttk.Frame(root)
canvas_frame.pack()

reset_plot()
update_feature_options()

root.mainloop()

# In[1]- 手動和自動
import tkinter as tk
from tkinter import ttk
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random


plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 修改中文字體
plt.rcParams['axes.unicode_minus'] = False # 顯示負號
# 全域變數
df = None
df_scaled = None
X = None
assignments = None
centers = None
iteration = 0
k = 3
canvas = None
ax = None
animation_running = False
feature_names = []
status_label = None  # 顯示狀態訊息

def update_feature_options(event=None):
    global df, df_scaled, feature_names

    dataset = dataset_var.get()
    if dataset == 'iris':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
    else:
        penguins = sns.load_dataset("penguins").dropna()
        df = penguins.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    feature_names = list(df_scaled.columns)
    feature1_menu['values'] = feature_names
    feature2_menu['values'] = feature_names

    feature1_var.set(feature_names[0])
    feature2_var.set(feature_names[1])

    update_preview()

def update_preview(event=None):
    global X, ax

    if not feature1_var.get() or not feature2_var.get():
        return

    f1, f2 = feature1_var.get(), feature2_var.get()
    X = df_scaled[[f1, f2]].values

    ax.clear()
    # 🔵 初始化改為藍色
    df_scaled.plot.scatter(x=f1, y=f2, ax=ax, color='blue', alpha=0.6, title="原始資料（未分群）")
    canvas.draw()
    status_label.config(text="請選擇模式並開始分群")

def reset_plot():
    global canvas, ax
    fig, ax = plt.subplots(figsize=(5, 5))
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack()
    canvas.draw()

def init_random_centers():
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return np.array([np.random.uniform(mins, maxs) for _ in range(k)])

def start_clustering():
    global X, centers, assignments, iteration, animation_running, k

    if animation_running:
        return

    k = int(k_var.get())
    f1, f2 = feature1_var.get(), feature2_var.get()
    X = df_scaled[[f1, f2]].values
    centers = init_random_centers()
    assignments = np.zeros(len(X))
    iteration = 0

    if mode_var.get() == "自動":
        animation_running = True
        animate_step()
    else:
        update_plot(assignments, centers)
        status_label.config(text="手動模式：請按『下一步』")

def step_once():
    global X, centers, assignments, iteration, animation_running

    if centers is None:
        return

    new_assignments = np.array([
        np.argmin([np.linalg.norm(x - c) for c in centers])
        for x in X
    ])

    new_centers = np.array([
        X[new_assignments == i].mean(axis=0) if np.any(new_assignments == i) else centers[i]
        for i in range(k)
    ])

    changed = not np.array_equal(new_assignments, assignments)
    update_plot(new_assignments, new_centers)

    assignments[:] = new_assignments
    centers[:] = new_centers
    iteration += 1

    if not changed:
        status_label.config(text="✅ 分群已收斂")
        print("✅ 分群已收斂")
    else:
        status_label.config(text=f"第 {iteration} 步：尚未收斂")

    return changed

def animate_step():
    global animation_running
    changed = step_once()

    if changed:
        root.after(800, animate_step)
    else:
        animation_running = False

def update_plot(assignments, centers):
    ax.clear()

    f1, f2 = feature1_var.get(), feature2_var.get()
    df_plot = df_scaled[[f1, f2]].copy()
    df_plot['cluster'] = assignments

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(k):
        cluster_df = df_plot[df_plot['cluster'] == i]
        cluster_df.plot.scatter(x=f1, y=f2, ax=ax, color=colors[i % len(colors)], label=f'群 {i}')
        center_x, center_y = centers[i]
        ax.scatter(center_x, center_y, s=200, color=colors[i % len(colors)], marker='X', edgecolors='black', linewidths=2)

    ax.set_title(f"KMeans Iteration {iteration}（模式：{mode_var.get()}）")
    canvas.draw()

# --- Tkinter UI 設定 ---
root = tk.Tk()
root.title("KMeans 分群動畫（手動/自動 + 隨機初始化 + 狀態顯示）")

dataset_var = tk.StringVar(value='iris')
feature1_var = tk.StringVar()
feature2_var = tk.StringVar()
k_var = tk.StringVar(value='3')
mode_var = tk.StringVar(value="自動")

frame = ttk.Frame(root)
frame.pack(padx=10, pady=10)

ttk.Label(frame, text="資料集：").grid(row=0, column=0)
dataset_menu = ttk.Combobox(frame, textvariable=dataset_var, values=["iris", "penguins"], state='readonly', width=10)
dataset_menu.grid(row=0, column=1)
dataset_menu.bind("<<ComboboxSelected>>", update_feature_options)

ttk.Label(frame, text="Feature 1：").grid(row=1, column=0)
feature1_menu = ttk.Combobox(frame, textvariable=feature1_var, state='readonly', width=15)
feature1_menu.grid(row=1, column=1)
feature1_menu.bind("<<ComboboxSelected>>", update_preview)

ttk.Label(frame, text="Feature 2：").grid(row=2, column=0)
feature2_menu = ttk.Combobox(frame, textvariable=feature2_var, state='readonly', width=15)
feature2_menu.grid(row=2, column=1)
feature2_menu.bind("<<ComboboxSelected>>", update_preview)

ttk.Label(frame, text="分群數 (k)：").grid(row=3, column=0)
ttk.Combobox(frame, textvariable=k_var, values=["2", "3", "4", "5"], state='readonly', width=5).grid(row=3, column=1)

ttk.Label(frame, text="模式：").grid(row=4, column=0)
ttk.Combobox(frame, textvariable=mode_var, values=["自動", "手動"], state='readonly', width=10).grid(row=4, column=1)

ttk.Button(frame, text="開始分群", command=start_clustering).grid(row=5, column=0, columnspan=2, pady=5)
ttk.Button(frame, text="下一步", command=step_once).grid(row=6, column=0, columnspan=2, pady=5)

status_label = ttk.Label(root, text="", foreground="blue")
status_label.pack(pady=5)

canvas_frame = ttk.Frame(root)
canvas_frame.pack()

reset_plot()
update_feature_options()

root.mainloop()

# %%
