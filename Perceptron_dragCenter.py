# In[]
# perceptron_gui_drag.py

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

# ✅ 中文支援
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# === 初始參數 ===
w = np.zeros(2)
b = 0
step = 0
learning_rate = 1.0
animation_started = False
animation_running = True
updates_in_current_epoch = 0
samples_seen = 0
interval_ms = 1000

# 正負類中心點（可拖曳）
pos_center = np.array([4, 4])
neg_center = np.array([1, 1])

# 資料
X = np.empty((0, 2))
y = np.array([])

# === Tkinter 主介面 ===
root = tk.Tk()
root.title("Perceptron 學習動畫")

# === 建立 matplotlib 畫布 ===
fig, ax = plt.subplots(figsize=(6, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
ax.set_xlim(-6, 10)
ax.set_ylim(-6, 10)
ax.set_title("Perceptron 學習動畫")

scat_pos = ax.scatter([], [], c='blue', label='Positive')
scat_neg = ax.scatter([], [], c='red', marker='x', label='Negative')
line, = ax.plot([], [], 'k--', label='Decision Boundary')
highlight, = ax.plot([], [], 'ro', markerfacecolor='none', linestyle='--', markersize=15, label='Current Input')
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend(loc='lower left')

# 拖曳控制點顯示
pos_dot, = ax.plot([pos_center[0]], [pos_center[1]], 'bo', markersize=10, label='+中心', picker=5)
neg_dot, = ax.plot([neg_center[0]], [neg_center[1]], 'rx', markersize=10, label='-中心', picker=5)
dragging_point = None

# === 訊息顯示區 ===
output_box = tk.Text(root, height=8, width=80)
output_box.pack(pady=5)

def log_message(msg):
    output_box.insert(tk.END, msg + "\n")
    output_box.see(tk.END)

# === 資料產生 ===
def generate_data():
    global X, y, step, updates_in_current_epoch, samples_seen, w, b
    try:
        n_pos = int(entry_pos.get())
        n_neg = int(entry_neg.get())
    except ValueError:
        log_message("⚠️ 請輸入有效的整數")
        return

    np.random.seed(42)
    X_pos = np.random.normal(loc=pos_center, scale=1.0, size=(n_pos, 2))
    X_neg = np.random.normal(loc=neg_center, scale=1.0, size=(n_neg, 2))
    y_pos = np.ones(n_pos)
    y_neg = -np.ones(n_neg)

    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))

    step = 0
    updates_in_current_epoch = 0
    samples_seen = 0
    w = np.zeros(2)
    b = 0

    output_box.delete('1.0', tk.END)
    log_message("✅ 已重新產生資料")
    scat_pos.set_offsets(X[y == 1])
    scat_neg.set_offsets(X[y == -1])
    canvas.draw()

# === 動畫更新函數 ===
def update(frame):
    global w, b, step, animation_running, updates_in_current_epoch, samples_seen, X, y
    if not animation_running or len(X) == 0:
        return

    xi = X[step]
    target = y[step]
    activation = np.dot(w, xi) + b

    samples_seen += 1

    if target * activation <= 0:
        updates_in_current_epoch += 1
        old_w = w.copy()
        old_b = b
        w += learning_rate * target * xi
        b += learning_rate * target

        print("❌ 錯誤分類！")
        print(f"x = {xi}, y = {target}")
        print(f"w: {np.round(old_w, 3).tolist()} → {np.round(w, 3).tolist()}, b: {old_b:.3f} → {b:.3f}")
        print(f"斜率:{-w[0]/w[1]:.3f}")
        print("-" * 40)

        log_message("❌ 錯誤分類！")
        log_message(f"x = {xi}, y = {target}")
        log_message(f"w: {np.round(old_w, 3).tolist()} → {np.round(w, 3).tolist()}, b: {old_b:.3f} → {b:.3f}")
        log_message(f"斜率:{-w[0]/w[1]:.3f}")
        log_message("-" * 40)

    scat_pos.set_offsets(X[y == 1])
    scat_neg.set_offsets(X[y == -1])
    highlight.set_data([xi[0]], [xi[1]])

    if w[1] != 0:
        x_vals = np.linspace(-6, 10, 200)
        y_vals = -(w[0] * x_vals + b) / w[1]
        line.set_data(x_vals, y_vals)
    else:
        line.set_data([], [])

    text.set_text(f"步驟：{frame+1}\nw: {np.round(w, 3).tolist()}\nb: {b:.3f}")
    canvas.draw()

    step = (step + 1) % len(X)
    if samples_seen >= len(X):
        if updates_in_current_epoch == 0:
            ani.event_source.stop()
            log_message("✅ 已收斂，動畫停止")
        updates_in_current_epoch = 0
        samples_seen = 0

# === 學習率控制 ===
frame_lr = ttk.Frame(root)
frame_lr.pack()

ttk.Label(frame_lr, text="學習率：").pack(side='left')
lr_value_label = ttk.Label(frame_lr, text="1.00")
lr_value_label.pack(side='right')

def on_lr_change(val):
    global learning_rate
    learning_rate = float(val)
    lr_value_label.config(text=f"{learning_rate:.2f}")

lr_slider = ttk.Scale(frame_lr, from_=0.01, to=2.0, orient='horizontal', command=on_lr_change, length=200)
lr_slider.set(1.0)
lr_slider.pack(side='left', padx=5)

# === 動畫速度滑桿 ===
frame_speed = ttk.Frame(root)
frame_speed.pack(pady=5)

ttk.Label(frame_speed, text="動畫間隔（秒）：").pack(side='left')
interval_value_label = ttk.Label(frame_speed, text="1.0 秒")
interval_value_label.pack(side='left', padx=5)

def on_interval_change(val):
    global interval_ms
    try:
        interval_ms = int(float(val) * 1000)
        interval_value_label.config(text=f"{float(val):.1f} 秒")
    except (tk.TclError, RuntimeError):
        pass

interval_slider = ttk.Scale(frame_speed, from_=0.1, to=2.0, orient='horizontal', command=on_interval_change, length=200)
interval_slider.set(1.0)
interval_slider.pack(side='left', padx=5)

# === 控制按鈕區 ===
frame_inputs = ttk.Frame(root)
frame_inputs.pack(pady=5)

ttk.Label(frame_inputs, text="+類數量：").pack(side='left')
entry_pos = ttk.Entry(frame_inputs, width=5)
entry_pos.insert(0, "10")
entry_pos.pack(side='left', padx=5)

ttk.Label(frame_inputs, text="-類數量：").pack(side='left')
entry_neg = ttk.Entry(frame_inputs, width=5)
entry_neg.insert(0, "10")
entry_neg.pack(side='left', padx=5)

ttk.Button(frame_inputs, text="產生資料", command=generate_data).pack(side='left', padx=5)

def start_animation():
    global ani, animation_started
    if not animation_started:
        start_btn.config(state='disabled')
        animation_started = True

        def start_anim():
            global ani
            ani = FuncAnimation(fig, update, frames=1000, interval=interval_ms, repeat=False)
            canvas.draw_idle()

        root.after(100, start_anim)

def toggle_animation():
    global animation_running
    animation_running = not animation_running
    toggle_btn.config(text="繼續" if not animation_running else "暫停")

def reset():
    global w, b, step, animation_started, updates_in_current_epoch, samples_seen, animation_running, X, y
    w = np.zeros(2)
    b = 0
    step = 0
    updates_in_current_epoch = 0
    samples_seen = 0
    animation_running = True
    animation_started = False
    X = np.empty((0, 2))
    y = np.array([])

    output_box.delete('1.0', tk.END)
    line.set_data([], [])
    highlight.set_data([], [])
    text.set_text("")
    scat_pos.set_offsets(np.empty((0, 2)))
    scat_neg.set_offsets(np.empty((0, 2)))
    canvas.draw()
    start_btn.config(state='normal')

start_btn = ttk.Button(root, text="開始動畫", command=start_animation)
start_btn.pack(pady=5)

toggle_btn = ttk.Button(root, text="暫停", command=toggle_animation)
toggle_btn.pack(pady=5)

clear_btn = ttk.Button(root, text="清除畫面", command=reset)
clear_btn.pack(pady=5)

exit_btn = ttk.Button(root, text="關閉程式", command=root.destroy)
exit_btn.pack(pady=5)

# === 拖曳控制事件 ===
def on_pick(event):
    global dragging_point
    if event.artist == pos_dot:
        dragging_point = 'pos'
    elif event.artist == neg_dot:
        dragging_point = 'neg'

def on_motion(event):
    if event.xdata is None or event.ydata is None:
        return
    if dragging_point == 'pos':
        pos_center[:] = [event.xdata, event.ydata]
        pos_dot.set_data([pos_center[0]], [pos_center[1]])
        canvas.draw()
    elif dragging_point == 'neg':
        neg_center[:] = [event.xdata, event.ydata]
        neg_dot.set_data([neg_center[0]], [neg_center[1]])
        canvas.draw()

def on_release(event):
    global dragging_point
    dragging_point = None

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.close(fig)
root.focus_force()
root.mainloop()

# In[]-訓練準確率/錯誤率圖表
# perceptron_gui_drag.py
# perceptron_gui_drag.py

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

# ✅ 中文支援
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# === 初始參數 ===
w = np.zeros(2)
b = 0
step = 0
learning_rate = 1.0
animation_started = False
animation_running = True
updates_in_current_epoch = 0
samples_seen = 0
interval_ms = 1000

# 正負類中心點（可拖曳）
pos_center = np.array([4, 4])
neg_center = np.array([1, 1])

# 資料
X = np.empty((0, 2))
y = np.array([])

accuracy_history = []
epoch_count = []

# === Tkinter 主介面 ===
root = tk.Tk()
root.title("Perceptron 學習動畫")

# === 建立 matplotlib 畫布（主圖） ===
fig, ax = plt.subplots(figsize=(6, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
ax.set_xlim(-6, 10)
ax.set_ylim(-6, 10)
ax.set_title("Perceptron 學習動畫")

scat_pos = ax.scatter([], [], c='blue', label='Positive')
scat_neg = ax.scatter([], [], c='red', marker='x', label='Negative')
line, = ax.plot([], [], 'k--', label='Decision Boundary')
highlight, = ax.plot([], [], 'ro', markerfacecolor='none', linestyle='--', markersize=15, label='Current Input')
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend(loc='lower left')

# === 訓練準確率圖 ===
# fig_acc, ax_acc = plt.subplots(figsize=(5, 2))
# canvas_acc = FigureCanvasTkAgg(fig_acc, master=root)
# canvas_acc.get_tk_widget().pack()
# ax_acc.set_title("準確率變化圖")
# ax_acc.set_xlabel("Epoch")
# ax_acc.set_ylabel("Accuracy")
# acc_line, = ax_acc.plot([], [], 'g.-')

# 拖曳控制點顯示
pos_dot, = ax.plot([pos_center[0]], [pos_center[1]], 'bo', markersize=10, label='+中心', picker=5)
neg_dot, = ax.plot([neg_center[0]], [neg_center[1]], 'rx', markersize=10, label='-中心', picker=5)
dragging_point = None

# === 訊息顯示區 ===
output_box = tk.Text(root, height=8, width=80)
output_box.pack(pady=5)

def log_message(msg):
    output_box.insert(tk.END, msg + "\n")
    output_box.see(tk.END)

# === 資料產生 ===
def generate_data():
    global X, y, step, updates_in_current_epoch, samples_seen, w, b, accuracy_history, epoch_count
    try:
        n_pos = int(entry_pos.get())
        n_neg = int(entry_neg.get())
    except ValueError:
        log_message("⚠️ 請輸入有效的整數")
        return

    np.random.seed(42)
    X_pos = np.random.normal(loc=pos_center, scale=1.0, size=(n_pos, 2))
    X_neg = np.random.normal(loc=neg_center, scale=1.0, size=(n_neg, 2))
    y_pos = np.ones(n_pos)
    y_neg = -np.ones(n_neg)

    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))

    step = 0
    updates_in_current_epoch = 0
    samples_seen = 0
    w = np.zeros(2)
    b = 0

    accuracy_history = []
    epoch_count = []

    output_box.delete('1.0', tk.END)
    log_message("✅ 已重新產生資料")
    scat_pos.set_offsets(X[y == 1])
    scat_neg.set_offsets(X[y == -1])
    # ax_acc.clear()
    # ax_acc.set_title("準確率變化圖")
    # ax_acc.set_xlabel("Epoch")
    # ax_acc.set_ylabel("Accuracy")
    canvas.draw()
    canvas_acc.draw()

# === 計算準確率 ===
def compute_accuracy():
    predictions = np.sign(np.dot(X, w) + b)
    correct = np.sum(predictions == y)
    return correct / len(y)

# === 動畫更新函數 ===
def update(frame):
    global w, b, step, animation_running, updates_in_current_epoch, samples_seen, X, y
    if not animation_running or len(X) == 0:
        return

    xi = X[step]
    target = y[step]
    activation = np.dot(w, xi) + b

    samples_seen += 1

    if target * activation <= 0:
        updates_in_current_epoch += 1
        old_w = w.copy()
        old_b = b
        w += learning_rate * target * xi
        b += learning_rate * target

        print("❌ 錯誤分類！")
        print(f"x = {xi}, y = {target}")
        print(f"w: {np.round(old_w, 3).tolist()} → {np.round(w, 3).tolist()}, b: {old_b:.3f} → {b:.3f}")
        print("-" * 40)

        log_message("❌ 錯誤分類！")
        log_message(f"x = {xi}, y = {target}")
        log_message(f"w: {np.round(old_w, 3).tolist()} → {np.round(w, 3).tolist()}, b: {old_b:.3f} → {b:.3f}")
        log_message("-" * 40)

    scat_pos.set_offsets(X[y == 1])
    scat_neg.set_offsets(X[y == -1])
    highlight.set_data([xi[0]], [xi[1]])

    if w[1] != 0:
        x_vals = np.linspace(-6, 10, 200)
        y_vals = -(w[0] * x_vals + b) / w[1]
        line.set_data(x_vals, y_vals)
    else:
        line.set_data([], [])

    text.set_text(f"步驟：{frame+1}\nw: {np.round(w, 3).tolist()}\nb: {b:.3f}")

    # 更新準確率圖
    if samples_seen >= len(X):
        acc = compute_accuracy()
        accuracy_history.append(acc)
        epoch_count.append(len(accuracy_history))
        ax_acc.clear()
        ax_acc.set_title("準確率變化圖")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        acc_line, = ax_acc.plot(epoch_count, accuracy_history, 'g.-')
        canvas_acc.draw()

    canvas.draw()

    step = (step + 1) % len(X)
    if samples_seen >= len(X):
        if updates_in_current_epoch == 0:
            ani.event_source.stop()
            log_message("✅ 已收斂，動畫停止")
        updates_in_current_epoch = 0
        samples_seen = 0

# === 拖曳控制事件 ===
def on_pick(event):
    global dragging_point
    if event.artist == pos_dot:
        dragging_point = 'pos'
    elif event.artist == neg_dot:
        dragging_point = 'neg'

def on_motion(event):
    if event.xdata is None or event.ydata is None:
        return
    if dragging_point == 'pos':
        pos_center[:] = [event.xdata, event.ydata]
        pos_dot.set_data([pos_center[0]], [pos_center[1]])
        canvas.draw()
    elif dragging_point == 'neg':
        neg_center[:] = [event.xdata, event.ydata]
        neg_dot.set_data([neg_center[0]], [neg_center[1]])
        canvas.draw()

def on_release(event):
    global dragging_point
    dragging_point = None

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# === 學習率控制 ===
frame_lr = ttk.Frame(root)
frame_lr.pack()

ttk.Label(frame_lr, text="學習率：").pack(side='left')
lr_value_label = ttk.Label(frame_lr, text="1.00")
lr_value_label.pack(side='right')

def on_lr_change(val):
    global learning_rate
    learning_rate = float(val)
    lr_value_label.config(text=f"{learning_rate:.2f}")

lr_slider = ttk.Scale(frame_lr, from_=0.01, to=2.0, orient='horizontal', command=on_lr_change, length=200)
lr_slider.set(1.0)
lr_slider.pack(side='left', padx=5)

# === 動畫速度滑桿 ===
frame_speed = ttk.Frame(root)
frame_speed.pack(pady=5)

ttk.Label(frame_speed, text="動畫間隔（秒）：").pack(side='left')
interval_value_label = ttk.Label(frame_speed, text="1.0 秒")
interval_value_label.pack(side='left', padx=5)

def on_interval_change(val):
    global interval_ms
    try:
        interval_ms = int(float(val) * 1000)
        interval_value_label.config(text=f"{float(val):.1f} 秒")
    except (tk.TclError, RuntimeError):
        pass

interval_slider = ttk.Scale(frame_speed, from_=0.1, to=2.0, orient='horizontal', command=on_interval_change, length=200)
interval_slider.set(1.0)
interval_slider.pack(side='left', padx=5)

# === 控制按鈕區 ===
frame_inputs = ttk.Frame(root)
frame_inputs.pack(pady=5)

ttk.Label(frame_inputs, text="+類數量：").pack(side='left')
entry_pos = ttk.Entry(frame_inputs, width=5)
entry_pos.insert(0, "10")
entry_pos.pack(side='left', padx=5)

ttk.Label(frame_inputs, text="-類數量：").pack(side='left')
entry_neg = ttk.Entry(frame_inputs, width=5)
entry_neg.insert(0, "10")
entry_neg.pack(side='left', padx=5)

ttk.Button(frame_inputs, text="產生資料", command=generate_data).pack(side='left', padx=5)

# === 開始/暫停/清除/關閉 ===
def start_animation():
    global ani, animation_started
    if not animation_started:
        start_btn.config(state='disabled')
        animation_started = True

        def start_anim():
            global ani
            ani = FuncAnimation(fig, update, frames=1000, interval=interval_ms, repeat=False)
            canvas.draw_idle()

        root.after(100, start_anim)

def toggle_animation():
    global animation_running
    animation_running = not animation_running
    toggle_btn.config(text="繼續" if not animation_running else "暫停")

def reset():
    global w, b, step, animation_started, updates_in_current_epoch, samples_seen, animation_running, X, y, accuracy_history, epoch_count
    w = np.zeros(2)
    b = 0
    step = 0
    updates_in_current_epoch = 0
    samples_seen = 0
    animation_running = True
    animation_started = False
    X = np.empty((0, 2))
    y = np.array([])
    accuracy_history = []
    epoch_count = []

    output_box.delete('1.0', tk.END)
    line.set_data([], [])
    highlight.set_data([], [])
    text.set_text("")
    scat_pos.set_offsets(np.empty((0, 2)))
    scat_neg.set_offsets(np.empty((0, 2)))
    ax_acc.clear()
    ax_acc.set_title("準確率變化圖")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    canvas.draw()
    canvas_acc.draw()
    start_btn.config(state='normal')

start_btn = ttk.Button(root, text="開始動畫", command=start_animation)
start_btn.pack(pady=5)

toggle_btn = ttk.Button(root, text="暫停", command=toggle_animation)
toggle_btn.pack(pady=5)

clear_btn = ttk.Button(root, text="清除畫面", command=reset)
clear_btn.pack(pady=5)

exit_btn = ttk.Button(root, text="關閉程式", command=root.destroy)
exit_btn.pack(pady=5)

plt.close(fig)
plt.close(fig_acc)
root.focus_force()
root.mainloop()

# In[]-要修改的錯誤版本
# perceptron_gui_drag.py

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始參數與資料
w = np.zeros(2)
b = 0
step = 0
learning_rate = 1.0
animation_started = False
animation_running = True
updates_in_current_epoch = 0
samples_seen = 0
interval_ms = 1000

pos_center = np.array([4, 4])
neg_center = np.array([1, 1])
X = np.empty((0, 2))
y = np.array([])
accuracy_history = []
epoch_count = []

root = tk.Tk()
root.title("Perceptron 學習動畫")

# 畫布與圖表
fig, ax = plt.subplots(figsize=(6, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
ax.set_xlim(-6, 10)
ax.set_ylim(-6, 10)
ax.set_title("Perceptron 學習動畫")
scat_pos = ax.scatter([], [], c='blue', label='Positive')
scat_neg = ax.scatter([], [], c='red', marker='x', label='Negative')
line, = ax.plot([], [], 'k--', label='Decision Boundary')
highlight, = ax.plot([], [], 'ro', markerfacecolor='none', linestyle='--', markersize=15, label='Current Input')
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend(loc='lower left')

fig_acc, ax_acc = plt.subplots(figsize=(5, 2))
canvas_acc = FigureCanvasTkAgg(fig_acc, master=root)
canvas_acc.get_tk_widget().pack()
ax_acc.set_title("準確率變化圖")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")

pos_dot, = ax.plot([pos_center[0]], [pos_center[1]], 'bo', markersize=10, label='+中心', picker=5)
neg_dot, = ax.plot([neg_center[0]], [neg_center[1]], 'rx', markersize=10, label='-中心', picker=5)
dragging_point = None

output_box = tk.Text(root, height=8, width=80)
output_box.pack(pady=5)

def log_message(msg):
    output_box.insert(tk.END, msg + "\n")
    output_box.see(tk.END)

def generate_data():
    global X, y, step, updates_in_current_epoch, samples_seen, w, b, accuracy_history, epoch_count
    try:
        n_pos = int(entry_pos.get())
        n_neg = int(entry_neg.get())
    except ValueError:
        log_message("⚠️ 請輸入有效的整數")
        return
    np.random.seed(42)
    X_pos = np.random.normal(loc=pos_center, scale=1.0, size=(n_pos, 2))
    X_neg = np.random.normal(loc=neg_center, scale=1.0, size=(n_neg, 2))
    y_pos = np.ones(n_pos)
    y_neg = -np.ones(n_neg)
    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    step = 0
    updates_in_current_epoch = 0
    samples_seen = 0
    w = np.zeros(2)
    b = 0
    accuracy_history = []
    epoch_count = []
    output_box.delete('1.0', tk.END)
    log_message("✅ 已重新產生資料")
    scat_pos.set_offsets(X[y == 1])
    scat_neg.set_offsets(X[y == -1])
    ax_acc.clear()
    ax_acc.set_title("準確率變化圖")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    canvas.draw()
    canvas_acc.draw()

def compute_accuracy():
    predictions = np.sign(np.dot(X, w) + b)
    correct = np.sum(predictions == y)
    return correct / len(y)

def update(frame):
    global w, b, step, animation_running, updates_in_current_epoch, samples_seen, X, y, ani
    if not animation_running or len(X) == 0:
        return

    xi = X[step]
    target = y[step]
    activation = np.dot(w, xi) + b

    if target * activation <= 0:
        updates_in_current_epoch += 1
        old_w = w.copy()
        old_b = b
        w += learning_rate * target * xi
        b += learning_rate * target

        print("❌ 錯誤分類！")
        print(f"x = {xi}, y = {target}")
        print(f"w: {np.round(old_w, 3).tolist()} → {np.round(w, 3).tolist()}, b: {old_b:.3f} → {b:.3f}")
        print("-" * 40)
        log_message("❌ 錯誤分類！")
        log_message(f"x = {xi}, y = {target}")
        log_message(f"w: {np.round(old_w, 3).tolist()} → {np.round(w, 3).tolist()}, b: {old_b:.3f} → {b:.3f}")
        log_message("-" * 40)

    scat_pos.set_offsets(X[y == 1])
    scat_neg.set_offsets(X[y == -1])
    highlight.set_data([xi[0]], [xi[1]])

    if w[1] != 0:
        x_vals = np.linspace(-6, 10, 200)
        y_vals = -(w[0] * x_vals + b) / w[1]
        line.set_data(x_vals, y_vals)
    else:
        line.set_data([], [])

    text.set_text(f"步驟：{samples_seen + 1},w: {np.round(w, 3).tolist()},b: {b:.3f}")
    canvas.draw()

    step += 1
    samples_seen += 1

    if step >= len(X):
        step = 0
        acc = compute_accuracy()
        accuracy_history.append(acc)
        epoch_count.append(len(accuracy_history))

        ax_acc.clear()
        ax_acc.set_title("準確率變化圖")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.plot(epoch_count, accuracy_history, 'g.-')
        canvas_acc.draw()

        if updates_in_current_epoch == 0:
            ani.event_source.stop()
            log_message("✅ 已收斂，動畫停止")
            animation_running = False

        updates_in_current_epoch = 0
        samples_seen = 0

    step += 1
    samples_seen += 1

    if step >= len(X):
        step = 0

        acc = compute_accuracy()
        accuracy_history.append(acc)
        epoch_count.append(len(accuracy_history))

        ax_acc.clear()
        ax_acc.set_title("準確率變化圖")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.plot(epoch_count, accuracy_history, 'g.-')
        canvas_acc.draw()

        if updates_in_current_epoch == 0:
            ani.event_source.stop()
            log_message("✅ 已收斂，動畫停止")
            animation_running = False

        updates_in_current_epoch = 0
        samples_seen = 0

    step += 1
    samples_seen += 1

    if step >= len(X):
        step = 0
        acc = compute_accuracy()
        accuracy_history.append(acc)
        epoch_count.append(len(accuracy_history))

        ax_acc.clear()
        ax_acc.set_title("準確率變化圖")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.plot(epoch_count, accuracy_history, 'g.-')
        canvas_acc.draw()

        if updates_in_current_epoch == 0:
            ani.event_source.stop()
            log_message("✅ 已收斂，動畫停止")
            animation_running = False

        updates_in_current_epoch = 0
        samples_seen = 0

    step += 1
    samples_seen += 1

    if step >= len(X):
        acc = compute_accuracy()
        accuracy_history.append(acc)
        epoch_count.append(len(accuracy_history))
        ax_acc.clear()
        ax_acc.set_title("準確率變化圖")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.plot(epoch_count, accuracy_history, 'g.-')
        canvas_acc.draw()

        if updates_in_current_epoch == 0:
            ani.event_source.stop()
            log_message("✅ 已收斂，動畫停止")
            animation_running = False

        step = 0
        updates_in_current_epoch = 0
        samples_seen = 0

    step = (step + 1) % len(X)
    samples_seen += 1

    if step == 0:
        acc = compute_accuracy()
        accuracy_history.append(acc)
        epoch_count.append(len(accuracy_history))
        ax_acc.clear()
        ax_acc.set_title("準確率變化圖")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.plot(epoch_count, accuracy_history, 'g.-')
        canvas_acc.draw()

        if updates_in_current_epoch == 0:
            ani.event_source.stop()
            log_message("✅ 已收斂，動畫停止")
            animation_running = False

        updates_in_current_epoch = 0
        samples_seen = 0

    step = (step + 1) % len(X)

    if step == 0:
        acc = compute_accuracy()
        accuracy_history.append(acc)
        epoch_count.append(len(accuracy_history))
        ax_acc.clear()
        ax_acc.set_title("準確率變化圖")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.plot(epoch_count, accuracy_history, 'g.-')
        canvas_acc.draw()

        if updates_in_current_epoch == 0:
            ani.event_source.stop()
            log_message("✅ 已收斂，動畫停止")
            animation_running = False

        updates_in_current_epoch = 0
        samples_seen = 0

    step = (step + 1) % len(X)

    if step == 0:
        acc = compute_accuracy()
        accuracy_history.append(acc)
        epoch_count.append(len(accuracy_history))
        ax_acc.clear()
        ax_acc.set_title("準確率變化圖")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.plot(epoch_count, accuracy_history, 'g.-')
        canvas_acc.draw()

        if updates_in_current_epoch == 0:
            ani.event_source.stop()
            log_message("✅ 已收斂，動畫停止")
            animation_running = False
        updates_in_current_epoch = 0
        samples_seen = 0
        acc = compute_accuracy()
        accuracy_history.append(acc)
        epoch_count.append(len(accuracy_history))
        ax_acc.clear()
        ax_acc.set_title("準確率變化圖")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.plot(epoch_count, accuracy_history, 'g.-')
        canvas_acc.draw()
    canvas.draw()
    step = (step + 1) % len(X)
    if samples_seen >= len(X):
        if updates_in_current_epoch == 0:
            ani.event_source.stop()
            log_message("✅ 已收斂，動畫停止")
        updates_in_current_epoch = 0
        samples_seen = 0

def on_pick(event):
    global dragging_point
    if event.artist == pos_dot:
        dragging_point = 'pos'
    elif event.artist == neg_dot:
        dragging_point = 'neg'

def on_motion(event):
    if event.xdata is None or event.ydata is None:
        return
    if dragging_point == 'pos':
        pos_center[:] = [event.xdata, event.ydata]
        pos_dot.set_data([pos_center[0]], [pos_center[1]])
        canvas.draw()
    elif dragging_point == 'neg':
        neg_center[:] = [event.xdata, event.ydata]
        neg_dot.set_data([neg_center[0]], [neg_center[1]])
        canvas.draw()

def on_release(event):
    global dragging_point
    dragging_point = None

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# === UI 控制元件 ===
frame_top = ttk.Frame(root)
frame_top.pack(pady=5)

frame_inputs = ttk.Labelframe(frame_top, text="資料產生")
frame_inputs.grid(row=0, column=0, padx=5, sticky="w")

ttk.Label(frame_inputs, text="+類數量：").grid(row=0, column=0)
entry_pos = ttk.Entry(frame_inputs, width=5)
entry_pos.insert(0, "10")
entry_pos.grid(row=0, column=1)

ttk.Label(frame_inputs, text="-類數量：").grid(row=0, column=2)
entry_neg = ttk.Entry(frame_inputs, width=5)
entry_neg.insert(0, "10")
entry_neg.grid(row=0, column=3)

ttk.Button(frame_inputs, text="產生資料", command=generate_data).grid(row=0, column=4, padx=5)

frame_param = ttk.Labelframe(frame_top, text="學習參數")
frame_param.grid(row=0, column=1, padx=10)

ttk.Label(frame_param, text="學習率：").grid(row=0, column=0)
lr_value_label = ttk.Label(frame_param, text="1.00")
lr_value_label.grid(row=0, column=2)

lr_slider = ttk.Scale(frame_param, from_=0.01, to=2.0, orient='horizontal', command=on_lr_change, length=150)
lr_slider.set(1.0)
lr_slider.grid(row=0, column=1)

ttk.Label(frame_param, text="間隔(秒)：").grid(row=1, column=0)
interval_value_label = ttk.Label(frame_param, text="1.0 秒")
interval_value_label.grid(row=1, column=2)

interval_slider = ttk.Scale(frame_param, from_=0.1, to=2.0, orient='horizontal', command=on_interval_change, length=150)
interval_slider.set(1.0)
interval_slider.grid(row=1, column=1)

frame_buttons = ttk.Labelframe(frame_top, text="控制")
frame_buttons.grid(row=0, column=2, padx=5)

start_btn = ttk.Button(frame_buttons, text="開始動畫", command=lambda: start_animation())
start_btn.grid(row=0, column=0, padx=5, pady=2)

toggle_btn = ttk.Button(frame_buttons, text="暫停", command=toggle_animation)
toggle_btn.grid(row=0, column=1, padx=5, pady=2)

clear_btn = ttk.Button(frame_buttons, text="清除畫面", command=reset)
clear_btn.grid(row=1, column=0, padx=5, pady=2)

exit_btn = ttk.Button(frame_buttons, text="關閉程式", command=root.destroy)
exit_btn.grid(row=1, column=1, padx=5, pady=2)

plt.close(fig)
plt.close(fig_acc)
root.focus_force()
root.mainloop()

# %%
