import numpy as np
import sounddevice as sd
import tkinter as tk
import threading
import time

# -------------------------------------------------
# AUDIO CONFIG
# -------------------------------------------------
sample_rate = 44100

# OPTIONAL: set this if needed after checking sd.query_devices()
OUTPUT_DEVICE = None   # e.g. 3

base_frequencies = {
    "1": 110.0,
    "2": 220.0,
    "3": 330.0,
    "4": 440.0
}

pitch_shift = 0.0
active_keys = set()
lock = threading.Lock()

# -------------------------------------------------
# LORENZ CIRCLE
# -------------------------------------------------
class LorenzCircle:
    def __init__(self):
        self.x = 0.1
        self.y = 0.0
        self.z = 0.0

        self.sigma = 10
        self.rho = 28
        self.beta = 8/3

        self.entropy = 0.0
        self.pulse = np.zeros(512)
        self.last_update = time.time()

    def step(self, dt):
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z

        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt

    def update_entropy(self):
        self.entropy = min(self.entropy + 0.02, 1.0)

        # entropy modifies Lorenz dynamics
        self.sigma = 10 + 6 * self.entropy
        self.rho = 28 + 20 * self.entropy
        self.beta = (8/3) + 2.5 * self.entropy

    def build_pulse(self):
        dt = 0.005
        samples = 400

        xs, ys, zs = [], [], []

        for _ in range(samples):
            self.step(dt)
            xs.append(self.x)
            ys.append(self.y)
            zs.append(self.z)

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        bottom = np.mean(xs)
        center = np.mean(ys)
        crest = np.mean(zs)

        nodes = np.array([bottom, center, crest])
        nodes -= np.mean(nodes)
        nodes /= np.max(np.abs(nodes)) + 1e-6

        pulse = np.zeros(512)
        third = len(pulse)//3

        pulse[:third] = nodes[0]
        pulse[third:2*third] = nodes[1]
        pulse[2*third:] = nodes[2]

        pulse = np.convolve(pulse, np.hanning(21), mode='same')

        pulse += self.entropy * 0.4 * np.power(pulse, 3)

        pulse /= np.max(np.abs(pulse)) + 1e-6

        self.pulse = pulse

    def maybe_update(self):
        if time.time() - self.last_update > 2:
            self.update_entropy()
            self.build_pulse()
            self.last_update = time.time()


circles = {k: LorenzCircle() for k in base_frequencies}

# -------------------------------------------------
# AUDIO CALLBACK
# -------------------------------------------------
def audio_callback(outdata, frames, time_info, status):
    outdata[:] = np.zeros((frames, 1))  # hard silence default

    with lock:
        keys = list(active_keys)
        shift = pitch_shift

    if not keys:
        return

    total_wave = np.zeros(frames)

    for key in keys:
        circle = circles[key]
        circle.maybe_update()

        freq = base_frequencies[key] * (2 ** shift)
        pulse = circle.pulse

        samples_per_cycle = int(sample_rate / freq)
        samples_per_cycle = max(16, samples_per_cycle)

        cycle = np.interp(
            np.linspace(0, len(pulse), samples_per_cycle, endpoint=False),
            np.arange(len(pulse)),
            pulse
        )

        repeats = int(np.ceil(frames / samples_per_cycle))
        wave = np.tile(cycle, repeats)[:frames]

        total_wave += wave

    total_wave /= len(keys)

    outdata[:] = total_wave.reshape(-1, 1)

# -------------------------------------------------
# START AUDIO STREAM
# -------------------------------------------------
stream = sd.OutputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback,
    device=OUTPUT_DEVICE
)

stream.start()
print("Audio stream started.")

# -------------------------------------------------
# TKINTER UI
# -------------------------------------------------
root = tk.Tk()
root.title("Drift-A-Tone")
root.geometry("600x250")
root.configure(bg="black")
root.focus_force()

canvas = tk.Canvas(root, width=600, height=200, bg="black", highlightthickness=0)
canvas.pack()

circle_ids = {}
positions = {"1":120,"2":240,"3":360,"4":480}

for k,x in positions.items():
    circle_ids[k] = canvas.create_oval(x-40,80,x+40,160,fill="gray")

def refresh_visuals():
    for k in base_frequencies:
        if k in active_keys:
            entropy = circles[k].entropy
            r = int(100 + entropy * 155)
            b = 255 - r
            color = f'#{r:02x}00{b:02x}'
            canvas.itemconfig(circle_ids[k], fill=color)
        else:
            canvas.itemconfig(circle_ids[k], fill="gray")

    root.after(50, refresh_visuals)

refresh_visuals()

# -------------------------------------------------
# KEY HANDLING
# -------------------------------------------------
def on_press(event):
    global pitch_shift

    key = event.char if event.char else event.keysym
    key = key.lower()

    with lock:
        if key in base_frequencies:
            active_keys.add(key)

        if key == "5" and active_keys:
            rightmost = max(active_keys)
            combined = np.zeros(512)
            for k in active_keys:
                combined += circles[k].pulse
            combined /= len(active_keys)
            circles[rightmost].pulse = combined

    if event.keysym == "Up":
        pitch_shift += 0.1
    if event.keysym == "Down":
        pitch_shift -= 0.1

def on_release(event):
    key = event.char if event.char else event.keysym
    key = key.lower()
    with lock:
        if key in active_keys:
            active_keys.remove(key)

root.bind("<KeyPress>", on_press)
root.bind("<KeyRelease>", on_release)

# -------------------------------------------------
# CLEAN SHUTDOWN
# -------------------------------------------------
def on_close():
    stream.stop()
    stream.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()


