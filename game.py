import pygame
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import numpy as np
import random
import sys
import threading
import time
import urllib.request
import os


SCREEN_W, SCREEN_H = 900, 500
FPS       = 60
GROUND_Y  = SCREEN_H - 80


WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
SKY    = (135, 206, 235)
GROUND = (83,  83,  83)
GREEN  = (34,  139, 34)
DINO_C = (80,  120, 60)
PIPE_C = (50,  168, 82)
RED    = (220, 50,  50)
ORANGE = (255, 140, 0)
YELLOW = (255, 220, 0)
GRAY   = (150, 150, 150)
DARK   = (40,  40,  40)
LBLUE  = (173, 216, 230)


DIFFICULTIES = {
    "Easy": {
        "speed": 4, "gravity": 0.4, "jump_force": -12,
        "spawn_min": 90, "spawn_max": 140, "lives": 5,
        "gap_min": 170, "gap_max": 220,
        "color": (50, 180, 50),
        "desc": "Slower speed · 5 lives · Wide gaps",
    },
    "Medium": {
        "speed": 5, "gravity": 0.5, "jump_force": -11,
        "spawn_min": 70, "spawn_max": 120, "lives": 3,
        "gap_min": 140, "gap_max": 190,
        "color": (220, 140, 0),
        "desc": "Normal speed · 3 lives · Normal gaps",
    },
    "Hard": {
        "speed": 7, "gravity": 0.65, "jump_force": -13,
        "spawn_min": 45, "spawn_max": 85, "lives": 1,
        "gap_min": 110, "gap_max": 150,
        "color": (220, 50, 50),
        "desc": "Fast speed · 1 life · Narrow gaps",
    },
}


MODEL_PATH = "hand_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading hand landmark model... please wait.")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("[INFO] Model downloaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not download model: {e}")
            print("[WARN] Running in keyboard-only mode.")
            return False
    return True



class HandGestureDetector:
    def __init__(self):
        self.gesture       = "NONE"
        self.hand_x        = 0.5
        self.hand_y        = 0.5
        self.running       = False
        self.lock          = threading.Lock()
        self.cam_available = False
        self.cap           = None
        self.landmarker    = None

    def start(self):
        model_ok = download_model()
        if not model_ok:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[WARN] No webcam found - keyboard-only mode.")
            return

        try:
            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            options = HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            self.landmarker    = mp_vision.HandLandmarker.create_from_options(options)
            self.cam_available = True
            self.running       = True
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()
            print("[INFO] Hand gesture detector started.")
        except Exception as e:
            print(f"[WARN] Could not start hand detector: {e}")
            print("[WARN] Running in keyboard-only mode.")

    def _fingers_up(self, landmarks, handedness):
        """Return list of 1/0 for each finger (thumb, index, middle, ring, pinky)."""
        tips = [4, 8, 12, 16, 20]
        fingers = []
      
        if handedness == "Right":
            fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
        else:
            fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
        
        for tip in tips[1:]:
            fingers.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)
        return fingers

    def _loop(self):
        prev_y = 0.5
        prev_x = 0.5
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result  = self.landmarker.detect(mp_img)

            gesture = "NONE"

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                lm          = result.hand_landmarks[0]   # first hand
                handedness  = "Right"
                if result.handedness and len(result.handedness) > 0:
                    handedness = result.handedness[0][0].category_name

                cx = lm[9].x   
                cy = lm[9].y

                fingers   = self._fingers_up(lm, handedness)
                total_up  = sum(fingers)

                if total_up == 0:
                    gesture = "FIST"
                else:
                    dy = cy - prev_y
                    dx = cx - prev_x
                    if abs(dy) > 0.018:
                        gesture = "DOWN" if dy > 0 else "UP"
                    elif abs(dx) > 0.03:
                        gesture = "LEFT" if dx < 0 else "RIGHT"

                prev_y = cy
                prev_x = cx

                with self.lock:
                    self.hand_x = cx
                    self.hand_y = cy

                
                for point in lm:
                    px = int(point.x * frame.shape[1])
                    py = int(point.y * frame.shape[0])
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

            with self.lock:
                self.gesture = gesture

            time.sleep(0.015)

    def get_gesture(self):
        with self.lock:
            return self.gesture

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.landmarker:
            self.landmarker.close()



class Dino:
    W, H   = 50, 60
    DUCK_H = 35

    def __init__(self, gravity, jump_force):
        self.gravity    = gravity
        self.jump_force = jump_force
        self.reset()

    def reset(self):
        self.x          = 120
        self.y          = GROUND_Y - self.H
        self.vy         = 0
        self.on_ground  = True
        self.ducking    = False
        self.jump_count = 0
        self.invincible = 0

    def jump(self):
        if self.jump_count < 2:
            self.vy         = self.jump_force
            self.on_ground  = False
            self.jump_count += 1

    def flap(self):
        self.vy        = self.jump_force * 0.8
        self.on_ground = False

    def duck(self, state):
        self.ducking = state

    def move_back(self):
        self.x = max(60, self.x - 3)

    def update(self):
        if self.invincible > 0:
            self.invincible -= 1
        h        = self.DUCK_H if self.ducking else self.H
        self.vy += self.gravity
        self.vy  = min(self.vy, 14)
        self.y  += self.vy
        ground   = GROUND_Y - h
        if self.y >= ground:
            self.y = ground; self.vy = 0
            self.on_ground = True; self.jump_count = 0
        else:
            self.on_ground = False
        if self.y < 0:
            self.y = 0; self.vy = 0

    @property
    def rect(self):
        h = self.DUCK_H if self.ducking else self.H
        return pygame.Rect(self.x, self.y, self.W, h)

    def draw(self, surface):
        h = self.DUCK_H if self.ducking else self.H
        if self.invincible > 0 and (self.invincible // 4) % 2:
            return
        pygame.draw.rect(surface, DINO_C, (self.x, self.y, self.W, h), border_radius=8)
        pygame.draw.circle(surface, WHITE, (self.x+36, self.y+12), 7)
        pygame.draw.circle(surface, BLACK, (self.x+38, self.y+13), 3)
        pygame.draw.line(surface, BLACK, (self.x+28, self.y+24), (self.x+44, self.y+22), 2)
        if not self.ducking:
            tick = pygame.time.get_ticks() // 150
            lo   = 8 if tick % 2 == 0 else -8
            pygame.draw.rect(surface, DINO_C, (self.x+10, self.y+h-14, 12, 14), border_radius=4)
            pygame.draw.rect(surface, DINO_C, (self.x+28, self.y+h-14+lo, 12, 14), border_radius=4)
            pygame.draw.polygon(surface, DINO_C, [
                (self.x, self.y+h//2), (self.x-18, self.y+h//2-8), (self.x-12, self.y+h//2+10)])
        pygame.draw.rect(surface, DINO_C, (self.x+36, self.y+h//2-4, 14, 10), border_radius=3)



class Obstacle:
    SPEED = 5

    def __init__(self, kind="cactus", gap_min=140, gap_max=200):
        self.kind = kind; self.passed = False
        if kind == "cactus":
            self.w = random.randint(22, 38); self.h = random.randint(50, 90)
            self.x = SCREEN_W + 20;         self.y = GROUND_Y - self.h
        else:
            gap        = random.randint(gap_min, gap_max)
            pipe_h     = random.randint(80, SCREEN_H - gap - 100)
            self.x     = SCREEN_W + 20; self.w = 60
            self.top_h = pipe_h
            self.bot_y = pipe_h + gap;  self.bot_h = SCREEN_H - self.bot_y
            self.h     = SCREEN_H

    def update(self):     self.x -= self.SPEED
    def off_screen(self): return self.x + (self.w if self.kind=="cactus" else 10) < 0

    def get_rects(self):
        if self.kind == "cactus":
            return [pygame.Rect(self.x, self.y, self.w, self.h)]
        return [pygame.Rect(self.x, 0, self.w, self.top_h),
                pygame.Rect(self.x, self.bot_y, self.w, self.bot_h)]

    def draw(self, surface):
        if self.kind == "cactus":
            pygame.draw.rect(surface, GREEN, (self.x, self.y, self.w, self.h), border_radius=6)
            ay = self.y + self.h//3
            pygame.draw.rect(surface, GREEN, (self.x-14, ay, 14, 16), border_radius=4)
            pygame.draw.rect(surface, GREEN, (self.x-14, ay-18, 10, 20), border_radius=4)
            ay2 = self.y + self.h//2
            pygame.draw.rect(surface, GREEN, (self.x+self.w, ay2, 16, 14), border_radius=4)
            pygame.draw.rect(surface, GREEN, (self.x+self.w+6, ay2-20, 10, 22), border_radius=4)
        else:
            pygame.draw.rect(surface, PIPE_C, (self.x, 0, self.w, self.top_h))
            pygame.draw.rect(surface, PIPE_C, (self.x-4, self.top_h-18, self.w+8, 18), border_radius=4)
            pygame.draw.rect(surface, PIPE_C, (self.x, self.bot_y, self.w, self.bot_h))
            pygame.draw.rect(surface, PIPE_C, (self.x-4, self.bot_y, self.w+8, 18), border_radius=4)
            pygame.draw.rect(surface, (100,210,120), (self.x+8, 0, 8, self.top_h))
            pygame.draw.rect(surface, (100,210,120), (self.x+8, self.bot_y, 8, self.bot_h))



class Coin:
    def __init__(self):
        self.x = SCREEN_W+20; self.y = random.randint(80, GROUND_Y-40)
        self.r = 10;          self.collected = False

    def update(self):     self.x -= Obstacle.SPEED
    def off_screen(self): return self.x < -20
    def rect(self):       return pygame.Rect(self.x-self.r, self.y-self.r, self.r*2, self.r*2)

    def draw(self, surface):
        pulse = abs((pygame.time.get_ticks() % 800) - 400) / 400
        r = int(self.r + pulse*3)
        pygame.draw.circle(surface, YELLOW, (int(self.x), int(self.y)), r)
        pygame.draw.circle(surface, ORANGE, (int(self.x), int(self.y)), r, 2)
        pygame.draw.circle(surface, WHITE,  (int(self.x)-3, int(self.y)-3), 3)



class Cloud:
    def __init__(self):
        self.x=SCREEN_W+random.randint(0,200); self.y=random.randint(40,160)
        self.w=random.randint(80,140);          self.speed=random.uniform(0.8,1.5)

    def update(self):     self.x -= self.speed
    def off_screen(self): return self.x+self.w < 0

    def draw(self, surface):
        pygame.draw.ellipse(surface, WHITE, (self.x, self.y, self.w, 30))
        pygame.draw.ellipse(surface, WHITE, (self.x+15, self.y-15, self.w-30, 28))
        pygame.draw.ellipse(surface, WHITE, (self.x+self.w-50, self.y-8, 50, 24))



class Particle:
    def __init__(self, x, y, color):
        self.x=x; self.y=y; self.color=color
        self.vx=random.uniform(-3,3); self.vy=random.uniform(-4,-1)
        self.life=random.randint(20,40); self.r=random.randint(3,7)

    def update(self):
        self.x+=self.vx; self.y+=self.vy; self.vy+=0.2; self.life-=1

    def draw(self, surface):
        r = max(1, self.r-(40-self.life)//10)
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), r)



def draw_text(surface, text, x, y, size=28, color=BLACK, bold=False):
    font = pygame.font.SysFont("Arial", size, bold=bold)
    surface.blit(font.render(text, True, color), (x, y))

def draw_heart(surface, x, y, size=18, filled=True):
    color = RED if filled else (180,180,180)
    r = size//2
    pygame.draw.circle(surface, color, (x+r//2, y+r//2), r//2)
    pygame.draw.circle(surface, color, (x+r+r//2, y+r//2), r//2)
    pygame.draw.polygon(surface, color, [(x, y+r//2),(x+r*2, y+r//2),(x+r, y+size)])

def draw_gesture_box(surface, gesture, cam_available):
    labels = {
        "UP":   ("UP - Jump / Fly",  (50,180,50)),
        "DOWN": ("DOWN - Duck",      (50,120,220)),
        "LEFT": ("LEFT - Back",      (220,150,50)),
        "FIST": ("FIST - Pause",     (200,50,50)),
        "NONE": ("No gesture",       GRAY),
    }
    label, color = labels.get(gesture, ("--", GRAY))
    status    = "CAM ON" if cam_available else "NO CAM - KB mode"
    cam_color = (50,180,50) if cam_available else RED
    pygame.draw.rect(surface, (230,230,240), (SCREEN_W-210,10,200,60), border_radius=8)
    pygame.draw.rect(surface, GRAY,          (SCREEN_W-210,10,200,60), 2, border_radius=8)
    draw_text(surface, status, SCREEN_W-200, 18, 14, cam_color, bold=True)
    draw_text(surface, label,  SCREEN_W-200, 38, 18, color,     bold=True)



class HandFlapDinoGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("HandFlap Dino Runner")
        self.clock  = pygame.time.Clock()

        self.detector = HandGestureDetector()
        self.detector.start()

        self.diff_names    = ["Easy", "Medium", "Hard"]
        self.selected_diff = 1
        self.difficulty    = "Medium"
        self.high_scores   = {k: 0 for k in DIFFICULTIES}
        self.state         = "menu"
        self.clouds        = [Cloud() for _ in range(4)]
        self.reset()

    def reset(self):
        cfg = DIFFICULTIES[self.difficulty]
        Obstacle.SPEED     = cfg["speed"]
        self.dino          = Dino(cfg["gravity"], cfg["jump_force"])
        self.obstacles     = []
        self.coins         = []
        self.particles     = []
        self.score         = 0.0
        self.lives         = cfg["lives"]
        self.max_lives     = cfg["lives"]
        self.speed_mult    = 1.0
        self.spawn_timer   = 0
        self.coin_timer    = 0
        self.cloud_timer   = 0
        self.distance      = 0
        self.mode          = "dino"
        self.mode_timer    = 0
        self.respawn_timer = 0
        self.gap_min       = cfg["gap_min"]
        self.gap_max       = cfg["gap_max"]

    def spawn_objects(self):
        cfg = DIFFICULTIES[self.difficulty]
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            kind = "cactus" if self.mode == "dino" else "pipe"
            self.obstacles.append(Obstacle(kind, self.gap_min, self.gap_max))
            gap = int(random.randint(cfg["spawn_min"], cfg["spawn_max"]) / self.speed_mult)
            self.spawn_timer = max(35, gap)
        self.coin_timer -= 1
        if self.coin_timer <= 0:
            self.coins.append(Coin()); self.coin_timer = random.randint(90, 160)
        self.cloud_timer -= 1
        if self.cloud_timer <= 0:
            self.clouds.append(Cloud()); self.cloud_timer = random.randint(60, 120)

    def handle_input(self):
        gesture = self.detector.get_gesture()
        keys    = pygame.key.get_pressed()
        if self.state != "playing" or self.respawn_timer > 0:
            return gesture
        if keys[pygame.K_UP] or keys[pygame.K_SPACE]:
            self.dino.flap() if self.mode=="flappy" else self.dino.jump()
        if keys[pygame.K_DOWN]: self.dino.duck(True)
        elif gesture != "DOWN": self.dino.duck(False)
        if keys[pygame.K_LEFT]: self.dino.move_back()
        if gesture == "UP":   self.dino.flap() if self.mode=="flappy" else self.dino.jump()
        elif gesture == "DOWN": self.dino.duck(True)
        elif gesture == "LEFT": self.dino.move_back()
        return gesture

    def check_collisions(self):
        if self.dino.invincible > 0 or self.respawn_timer > 0:
            return
        dr = self.dino.rect
        for obs in self.obstacles:
            for r in obs.get_rects():
                if dr.colliderect(r.inflate(-6,-6)):
                    self._lose_life(); return
        for coin in self.coins:
            if not coin.collected and dr.colliderect(coin.rect()):
                coin.collected = True; self.score += 5
                self._burst(coin.x, coin.y, YELLOW, 10)
        if self.mode=="flappy" and self.dino.y >= GROUND_Y-self.dino.H:
            self._lose_life()

    def _lose_life(self):
        self.lives -= 1
        self._burst(self.dino.x+25, self.dino.y+30, RED, 20)
        if self.lives <= 0:
            self.high_scores[self.difficulty] = max(
                self.high_scores[self.difficulty], int(self.score))
            self.state = "dead"
        else:
            self.dino.invincible = 120; self.respawn_timer = 60
            self.obstacles.clear();     self.spawn_timer = 90

    def _burst(self, x, y, color, n):
        for _ in range(n): self.particles.append(Particle(x, y, color))

    def update_mode(self):
        self.mode_timer += 1
        if self.mode_timer > 720:
            self.mode_timer = 0
            self.mode = "flappy" if self.mode=="dino" else "dino"
            self.obstacles.clear(); self.spawn_timer = 90
            self._burst(SCREEN_W//2, SCREEN_H//2, LBLUE, 25)

    def update(self):
        if self.state != "playing": return
        if self.respawn_timer > 0:
            self.respawn_timer -= 1
            for c in self.clouds: c.update()
            return
        self.distance   += 1
        self.score      += 0.05 * self.speed_mult
        self.speed_mult  = 1 + self.distance/3000
        Obstacle.SPEED   = int(DIFFICULTIES[self.difficulty]["speed"] * self.speed_mult)
        self.update_mode(); self.spawn_objects()
        self.dino.update()
        for obs in self.obstacles:
            obs.update()
            w = obs.w if obs.kind=="cactus" else 60
            if not obs.passed and obs.x+w < self.dino.x:
                obs.passed = True; self.score += 10
                self._burst(self.dino.x, self.dino.y, GREEN, 8)
        for c in self.coins:     c.update()
        for c in self.clouds:    c.update()
        for p in self.particles: p.update()
        self.obstacles  = [o for o in self.obstacles  if not o.off_screen()]
        self.coins      = [c for c in self.coins      if not c.off_screen() and not c.collected]
        self.clouds     = [c for c in self.clouds     if not c.off_screen()]
        self.particles  = [p for p in self.particles  if p.life > 0]
        self.check_collisions()

    
    def draw_bg(self):
        self.screen.fill(SKY)
        pygame.draw.circle(self.screen, YELLOW,        (80,60), 30)
        pygame.draw.circle(self.screen, (255,255,200), (80,60), 36, 3)
        for c in self.clouds: c.draw(self.screen)
        pygame.draw.rect(self.screen, GROUND,    (0, GROUND_Y, SCREEN_W, SCREEN_H-GROUND_Y))
        pygame.draw.rect(self.screen, (60,60,60),(0, GROUND_Y, SCREEN_W, 4))
        for i in range(0, SCREEN_W, 30):
            pygame.draw.circle(self.screen,(100,100,100),(i+(self.distance*2%30),GROUND_Y+15),2)

    def draw_hud(self, gesture):
        pygame.draw.rect(self.screen,(230,230,240),(10,10,200,56),border_radius=8)
        pygame.draw.rect(self.screen,GRAY,         (10,10,200,56),2,border_radius=8)
        draw_text(self.screen, f"Score: {int(self.score)}", 20, 18, 20, DARK)
        draw_text(self.screen, f"Best:  {self.high_scores[self.difficulty]}", 20, 40, 16, GRAY)
        pygame.draw.rect(self.screen,(230,230,240),(10,74,200,36),border_radius=8)
        pygame.draw.rect(self.screen,GRAY,         (10,74,200,36),2,border_radius=8)
        draw_text(self.screen,"Lives:",18,82,16,DARK,bold=True)
        for i in range(self.max_lives):
            draw_heart(self.screen, 72+i*26, 80, 18, i < self.lives)
        cfg = DIFFICULTIES[self.difficulty]
        pygame.draw.rect(self.screen, cfg["color"], (10,118,100,24), border_radius=6)
        draw_text(self.screen, self.difficulty, 22, 122, 15, WHITE, bold=True)
        mc = (50,150,220) if self.mode=="flappy" else (80,160,80)
        ml = "FLAPPY MODE" if self.mode=="flappy" else "DINO MODE"
        pygame.draw.rect(self.screen, mc, (SCREEN_W//2-80,8,160,28), border_radius=6)
        draw_text(self.screen, ml, SCREEN_W//2-68, 14, 16, WHITE, bold=True)
        bw = int(120*min(self.speed_mult/3,1))
        pygame.draw.rect(self.screen,(200,200,200),(10,150,120,10),border_radius=5)
        pygame.draw.rect(self.screen,ORANGE,       (10,150,bw, 10),border_radius=5)
        draw_text(self.screen,"Speed",140,146,14,GRAY)
        if self.respawn_timer > 0:
            draw_text(self.screen,"Respawning...",SCREEN_W//2-80,SCREEN_H//2-15,28,RED,bold=True)
        draw_gesture_box(self.screen, gesture, self.detector.cam_available)

    def draw_menu(self):
        self.screen.fill(SKY)
        for c in self.clouds: c.draw(self.screen); c.update()
        pygame.draw.rect(self.screen,GROUND,(0,GROUND_Y,SCREEN_W,SCREEN_H-GROUND_Y))
        draw_text(self.screen,"HandFlap Dino Runner",SCREEN_W//2-210,70,50,DARK,bold=True)
        draw_text(self.screen,"Hand Gesture Controlled Game",SCREEN_W//2-175,132,26,(80,80,80))
        pygame.draw.rect(self.screen,(50,150,220),(SCREEN_W//2-120,200,240,54),border_radius=10)
        draw_text(self.screen,"PLAY",SCREEN_W//2-34,214,28,WHITE,bold=True)
        pygame.draw.rect(self.screen,(100,100,100),(SCREEN_W//2-120,268,240,44),border_radius=10)
        draw_text(self.screen,"QUIT  (Q key)",SCREEN_W//2-78,280,20,WHITE)
        pygame.draw.rect(self.screen,(240,240,250),(SCREEN_W//2-240,330,480,118),border_radius=10)
        pygame.draw.rect(self.screen,GRAY,         (SCREEN_W//2-240,330,480,118),2,border_radius=10)
        hints=["UP arrow / Hand UP    =  Jump or Fly",
               "DOWN arrow / Hand DOWN  =  Duck or Drop",
               "LEFT arrow / Hand LEFT  =  Move backward",
               "Fist gesture / P key    =  Pause"]
        for i,h in enumerate(hints):
            draw_text(self.screen,h,SCREEN_W//2-220,340+i*26,16,DARK)
        if (pygame.time.get_ticks()//500)%2==0:
            draw_text(self.screen,"Press SPACE or ENTER to continue",
                      SCREEN_W//2-170,462,20,(50,120,220),bold=True)
        pygame.display.flip()

    def draw_difficulty(self):
        self.screen.fill(SKY)
        for c in self.clouds: c.draw(self.screen); c.update()
        pygame.draw.rect(self.screen,GROUND,(0,GROUND_Y,SCREEN_W,SCREEN_H-GROUND_Y))
        draw_text(self.screen,"Select Difficulty",SCREEN_W//2-165,30,44,DARK,bold=True)
        draw_text(self.screen,"Use LEFT / RIGHT arrows to choose, then press ENTER",
                  SCREEN_W//2-240,88,17,GRAY)
        for i,name in enumerate(self.diff_names):
            cfg=DIFFICULTIES[name]; bx=SCREEN_W//2-330+i*220; by=130; bw,bh=200,200
            selected=(i==self.selected_diff)
            pygame.draw.rect(self.screen,(180,180,190),(bx+4,by+4,bw,bh),border_radius=14)
            bg=cfg["color"] if selected else (230,230,240)
            pygame.draw.rect(self.screen,bg,(bx,by,bw,bh),border_radius=14)
            pygame.draw.rect(self.screen,WHITE if selected else GRAY,
                             (bx,by,bw,bh),3,border_radius=14)
            tc=WHITE if selected else DARK
            draw_text(self.screen,name,bx+bw//2-30,by+16,30,tc,bold=True)
            lives=cfg["lives"]; hx=bx+bw//2-(lives*24)//2
            for h in range(lives): draw_heart(self.screen,hx+h*26,by+60,20,True)
            for j,line in enumerate(cfg["desc"].split(" · ")):
                draw_text(self.screen,line,bx+10,by+108+j*30,14,tc)
            if selected:
                draw_text(self.screen,"SELECTED",bx+bw//2-42,by+bh-30,15,WHITE,bold=True)
        draw_text(self.screen,"Press ENTER or SPACE to Start",
                  SCREEN_W//2-158,352,22,(50,120,220),bold=True)
        draw_text(self.screen,"Press BACKSPACE to go back",SCREEN_W//2-120,384,17,GRAY)
        pygame.display.flip()

    def draw_game(self, gesture):
        self.draw_bg()
        for obs in self.obstacles: obs.draw(self.screen)
        for c   in self.coins:     c.draw(self.screen)
        self.dino.draw(self.screen)
        for p in self.particles:   p.draw(self.screen)
        self.draw_hud(gesture)
        if self.mode_timer < 90:
            alpha = int(180*(1-self.mode_timer/90))
            flash = pygame.Surface((SCREEN_W,SCREEN_H), pygame.SRCALPHA)
            flash.fill((173,216,230,alpha))
            self.screen.blit(flash,(0,0))
            msg = "FLAPPY SECTION!" if self.mode=="flappy" else "DINO SECTION!"
            draw_text(self.screen,msg,SCREEN_W//2-100,SCREEN_H//2-20,32,DARK,bold=True)
        pygame.display.flip()

    def draw_paused(self):
        ov=pygame.Surface((SCREEN_W,SCREEN_H),pygame.SRCALPHA); ov.fill((0,0,0,80))
        self.screen.blit(ov,(0,0))
        draw_text(self.screen,"PAUSED",SCREEN_W//2-72,SCREEN_H//2-24,48,WHITE,bold=True)
        draw_text(self.screen,"SPACE to resume  |  M for menu",SCREEN_W//2-162,SCREEN_H//2+36,22,LBLUE)
        pygame.display.flip()

    def draw_dead(self):
        ov=pygame.Surface((SCREEN_W,SCREEN_H),pygame.SRCALPHA); ov.fill((0,0,0,120))
        self.screen.blit(ov,(0,0))
        pygame.draw.rect(self.screen,WHITE,(SCREEN_W//2-200,SCREEN_H//2-130,400,260),border_radius=16)
        pygame.draw.rect(self.screen,RED,  (SCREEN_W//2-200,SCREEN_H//2-130,400,260),3,border_radius=16)
        draw_text(self.screen,"GAME OVER",SCREEN_W//2-110,SCREEN_H//2-115,42,RED,bold=True)
        draw_text(self.screen,f"Score:      {int(self.score)}",SCREEN_W//2-90,SCREEN_H//2-58,26,DARK)
        draw_text(self.screen,f"Best score: {self.high_scores[self.difficulty]}",SCREEN_W//2-90,SCREEN_H//2-22,22,GRAY)
        cfg=DIFFICULTIES[self.difficulty]
        draw_text(self.screen,f"Difficulty: {self.difficulty}",SCREEN_W//2-90,SCREEN_H//2+10,22,cfg["color"],bold=True)
        draw_text(self.screen,"Lives used:",SCREEN_W//2-90,SCREEN_H//2+40,18,DARK)
        for i in range(self.max_lives):
            draw_heart(self.screen,SCREEN_W//2+32+i*26,SCREEN_H//2+38,18,False)
        pygame.draw.rect(self.screen,(50,150,220),(SCREEN_W//2-150,SCREEN_H//2+76,130,44),border_radius=8)
        draw_text(self.screen,"Retry",SCREEN_W//2-118,SCREEN_H//2+86,24,WHITE,bold=True)
        pygame.draw.rect(self.screen,(100,100,100),(SCREEN_W//2+20,SCREEN_H//2+76,130,44),border_radius=8)
        draw_text(self.screen,"Menu",SCREEN_W//2+52,SCREEN_H//2+86,24,WHITE,bold=True)
        pygame.display.flip()

   
   
    def run(self):
        while True:
            gesture = self.detector.get_gesture()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.detector.stop(); pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.detector.stop(); pygame.quit(); sys.exit()
                    if self.state == "menu":
                        if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                            self.state = "difficulty"
                    elif self.state == "difficulty":
                        if event.key == pygame.K_LEFT:
                            self.selected_diff = max(0, self.selected_diff-1)
                        elif event.key == pygame.K_RIGHT:
                            self.selected_diff = min(2, self.selected_diff+1)
                        elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                            self.difficulty = self.diff_names[self.selected_diff]
                            self.reset(); self.state = "playing"
                        elif event.key == pygame.K_BACKSPACE:
                            self.state = "menu"
                    elif self.state == "playing":
                        if event.key == pygame.K_p: self.state = "paused"
                        if event.key in (pygame.K_UP, pygame.K_SPACE):
                            self.dino.flap() if self.mode=="flappy" else self.dino.jump()
                    elif self.state == "paused":
                        if event.key in (pygame.K_SPACE, pygame.K_p): self.state = "playing"
                        elif event.key == pygame.K_m: self.state = "menu"
                    elif self.state == "dead":
                        if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                            self.reset(); self.state = "playing"
                        elif event.key == pygame.K_m: self.state = "menu"

            if   self.state=="menu"    and gesture in ("UP","DOWN","LEFT"): self.state="difficulty"
            elif self.state=="playing" and gesture=="FIST":                 self.state="paused"
            elif self.state=="paused"  and gesture=="FIST":                 self.state="playing"
            elif self.state=="dead"    and gesture=="UP":
                self.reset(); self.state="playing"

            self.handle_input()
            self.update()

            if   self.state=="menu":       self.draw_menu()
            elif self.state=="difficulty": self.draw_difficulty()
            elif self.state=="playing":    self.draw_game(gesture)
            elif self.state=="paused":     self.draw_game(gesture); self.draw_paused()
            elif self.state=="dead":       self.draw_game(gesture); self.draw_dead()

            self.clock.tick(FPS)


if __name__ == "__main__":
    game = HandFlapDinoGame()
    game.run()