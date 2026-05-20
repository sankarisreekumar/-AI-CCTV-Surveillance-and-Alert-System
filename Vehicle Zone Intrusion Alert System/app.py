import tkinter as tk
from tkinter import filedialog
import threading

from alert import run_detection  # connect


def browse_video():
    path = filedialog.askopenfilename()
    entry_video.delete(0, tk.END)
    entry_video.insert(0, path)


def browse_model():
    path = filedialog.askopenfilename()
    entry_model.delete(0, tk.END)
    entry_model.insert(0, path)


def start_system():
    video = entry_video.get()
    model = entry_model.get()

    print("Starting Alert System...")

    # 🔥 Run detection without freezing GUI
    threading.Thread(target=run_detection, args=(video, model)).start()


# GUI
root = tk.Tk()
root.title("AI Alert System")
root.geometry("500x250")

tk.Label(root, text="Video Path / Camera URL").pack(pady=5)

frame_video = tk.Frame(root)
frame_video.pack()

entry_video = tk.Entry(frame_video, width=40)
entry_video.pack(side=tk.LEFT, padx=5)

tk.Button(frame_video, text="Browse", command=browse_video).pack(side=tk.LEFT)

tk.Label(root, text="Model Path").pack(pady=5)

frame_model = tk.Frame(root)
frame_model.pack()

entry_model = tk.Entry(frame_model, width=40)
entry_model.insert(0, "best.pt")
entry_model.pack(side=tk.LEFT, padx=5)

tk.Button(frame_model, text="Browse", command=browse_model).pack(side=tk.LEFT)

tk.Button(root, text="START", bg="green", fg="white",
          font=("Arial", 12), command=start_system).pack(pady=20)

print("✅ Interface Ready")

root.mainloop()
