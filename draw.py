import pygame
import tkinter as tk    
import numpy as np
from predict import load_params, predict

# ======================
# Config
# ======================
CANVAS_SIZE = 280
PANEL_WIDTH = 400
PANEL_HEIGHT = 320

GRID_SIZE = 28
SCALE = CANVAS_SIZE // GRID_SIZE
BRUSH_RADIUS = 8
AUTO_PREDICT_INTERVAL = 300  # ms

# ======================
# Init pygame
# ======================
pygame.init()
pygame.font.init()

canvas_screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))
pygame.display.set_caption("Draw a digit")

pygame.display.set_caption("Prediction")

canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
canvas.fill((0, 0, 0))

FONT_TITLE = pygame.font.SysFont("arial", 36, bold=True)
FONT_DIGIT = pygame.font.SysFont("arial", 28, bold=True)
FONT_PROB = pygame.font.SysFont("arial", 20)

# ======================
# Model + state
# ======================
params = load_params()

panel_visible = False
last_predict_time = 0
last_probs = None
last_pred = None

drawing = False
prev_pos = None

# ======================
# Drawing helpers
# ======================
def draw_smooth_line(start, end):
    dist = max(1, int(np.linalg.norm(np.array(end) - np.array(start))))
    for i in range(dist):
        t = i / dist
        x = int(start[0] + t * (end[0] - start[0]))
        y = int(start[1] + t * (end[1] - start[1]))
        pygame.draw.circle(canvas, (255, 255, 255), (x, y), BRUSH_RADIUS)

# ======================
# Preprocessing
# ======================
def process_input():
    pixels = pygame.surfarray.array3d(canvas)
    pixels = np.transpose(pixels, (1, 0, 2))

    gray = pixels[:, :, 0].astype(np.float32)

    gray = (
        gray +
        np.roll(gray, 1, axis=0) +
        np.roll(gray, -1, axis=0) +
        np.roll(gray, 1, axis=1) +
        np.roll(gray, -1, axis=1)
    ) / 5.0

    small = np.zeros((28, 28), dtype=np.float32)
    for i in range(28):
        for j in range(28):
            block = gray[i*SCALE:(i+1)*SCALE, j*SCALE:(j+1)*SCALE]
            small[i, j] = np.mean(block)

    small /= 255.0

    ys, xs = np.where(small > 0.05)
    if len(xs) > 0:
        small = np.roll(small, 14 - int(ys.mean()), axis=0)
        small = np.roll(small, 14 - int(xs.mean()), axis=1)

    return small.reshape(784, 1)

# ======================
# Prediction
# ======================
def predict_digit():
    global last_probs, last_pred

    X = process_input()
    if np.mean(X) < 0.01:
        return

    preds, probs = predict(X, params)
    last_probs = probs[:, 0]
    last_pred = preds[0]

# ======================
# Panel UI
# ======================
def draw_panel():
    panel_canvas.delete("all")

    if last_probs is None:
        panel_root.update()
        return

    top_idx = int(np.argmax(last_probs))
    top_prob = float(last_probs[top_idx])

    if top_prob >= 0.8:
        title = f"That's a {top_idx}!"
    elif top_prob >= 0.5:
        title = f"Is that a {top_idx}?"
    else:
        title = f"Maybe a {top_idx}?"

    panel_canvas.create_text(
        200, 30,
        text=title,
        font=("Arial", 24, "bold"),
        fill="black"
    )

    start_y = 80
    row_h = 40
    col_x = [120, 260]

    for d in range(10):
        col = d // 5
        row = d % 5
        x = col_x[col]
        y = start_y + row * row_h

        prob = last_probs[d] * 100
        fill = "#b4e6b4" if d == top_idx else "white"

        panel_canvas.create_oval(
            x-18, y-18, x+18, y+18,
            fill=fill, outline="black", width=2
        )

        panel_canvas.create_text(
            x, y,
            text=str(d),
            font=("Arial", 16, "bold")
        )

        panel_canvas.create_text(
            x + 45, y,
            text=f"{prob:.1f}%",
            font=("Arial", 12)
        )

    panel_root.update()

# ======================
# Tkinter panel window
# ======================
panel_root = tk.Tk()
panel_root.title("Prediction")
panel_root.geometry("400x320")
panel_root.resizable(False, False)

panel_canvas = tk.Canvas(panel_root, width=400, height=320, bg="#f5f5f5")
panel_canvas.pack()

# ======================
# Main loop
# ======================
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            prev_pos = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            prev_pos = None

        elif event.type == pygame.MOUSEMOTION and drawing:
            draw_smooth_line(prev_pos, event.pos)
            prev_pos = event.pos

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                panel_visible = True
                predict_digit()
                last_predict_time = pygame.time.get_ticks()

            elif event.key == pygame.K_c:
                canvas.fill((0, 0, 0))
                last_probs = None
                last_pred = None

    if panel_visible:
        now = pygame.time.get_ticks()
        if now - last_predict_time >= AUTO_PREDICT_INTERVAL:
            predict_digit()
            last_predict_time = now

    canvas_screen.blit(canvas, (0, 0))
    pygame.display.flip()

    if panel_visible:
        draw_panel()

    clock.tick(60)

pygame.quit()