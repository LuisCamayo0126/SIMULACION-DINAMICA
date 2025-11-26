"""
sim_game_simulator.py
Simulación estilo "videojuego" de la droguería MENAR usando SimPy + Pygame.

- Duración real: ~120 segundos (configurable)
- Duración simulada: configurable (por defecto 3600 s simulados → 1 h)
- Tiempo comprimido: los eventos se reproducen en "realtime" escalado
- Visual: clientes como círculos, fila a la izquierda, cajeros a la derecha
- Datos reales cargados desde Excel

Requisitos:
pip install simpy pygame numpy openpyxl
"""

import simpy
import random
import threading
import time
import math
import pygame
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import os

# ---------------------------
# CONFIGURACIÓN / PARÁMETROS
# ---------------------------

# Cargar datos reales desde Excel
excel_path = "datos_drogueria.xlsx"
cliente_data = {}  # client_id -> {"arrival": tiempo_interarribo, "service": tiempo}
arrival_is_absolute = False

try:
    df = pd.read_excel(excel_path)
    # Esperamos columnas: Cliente, Tiempo_Llegada, Tiempo_Servicio (o similar)
    arrivals = []
    services = []
    for idx, row in df.iterrows():
        # leer columnas por posición (ajustable)
        a = float(row.iloc[1]) if len(row) > 1 else 124.285714
        s = float(row.iloc[2]) if len(row) > 2 else 324.542857
        arrivals.append(max(0.0, a))
        services.append(max(1.0, s))

    # Detectar si las llegadas en Excel son tiempos absolutos (timestamps simulación)
    monotonic = all(arrivals[i] <= arrivals[i+1] for i in range(len(arrivals)-1)) if len(arrivals) > 1 else True
    if monotonic and len(arrivals) > 1:
        # si el máximo es mayor que la media de llegada, es probable que sean tiempos absolutos
        if max(arrivals) > 124.285714 * 1.5:
            arrival_is_absolute = True

    if arrival_is_absolute:
        # convertir absolutos a intervalos
        inters = []
        prev = 0.0
        for a in arrivals:
            inters.append(max(0.0, a - prev))
            prev = a
        arrivals = inters

    # poblar cliente_data con intervalos y servicios
    for idx in range(len(arrivals)):
        client_id = f"C{idx+1}"
        cliente_data[client_id] = {"arrival": max(0.0, arrivals[idx]), "service": max(1.0, services[idx])}

    print(f"✓ Cargados {len(cliente_data)} clientes desde Excel (arrival_is_absolute={arrival_is_absolute})")
except Exception as e:
    print(f"⚠ No se pudo cargar Excel ({e}), usando parámetros por defecto")

# Parámetros basados en tus datos reales:
MEAN_ARRIVAL = 124.285714        # segundos entre llegadas (media) - fallback
MEAN_SERVICE = 324.542857        # segundos servicio (media) - fallback
SD_SERVICE = 106.237864          # desviación estándar servicio

NUM_SERVERS = 6                  # cajeros observados en campo
SIM_SECONDS = 3600               # segundos simulados (ej: 3600 = 1 hora)
REAL_SECONDS = 60                # duración real esperada (aprox 1 minuto)
TIME_SCALE = SIM_SECONDS / REAL_SECONDS  # cuántos segundos sim corresponden a 1s real

# Visual / juego - MEJORADO CON COLORES GRADIENTES
WINDOW_W, WINDOW_H = 1000, 700
FPS = 60
CLIENT_RADIUS = 14
QUEUE_X = 150
QUEUE_Y = 140
QUEUE_SPACING = 40
SERVER_X = 340
SERVER_Y_START = 60
SERVER_SPACING = 80

# Colores gradientes y paleta mejorada
COLOR_BG = (12, 15, 25)          # fondo oscuro elegante
COLOR_PRIMARY = (100, 200, 255)   # azul primario
COLOR_ACCENT = (255, 150, 100)    # naranja acento
COLOR_SUCCESS = (100, 255, 150)   # verde éxito
COLOR_PANEL = (25, 35, 55)        # panel background
COLOR_TEXT = (240, 245, 255)      # texto principal

GRADIENT_COLORS = [
    (255, 100, 130),  # rojo-rosa
    (255, 150, 100),  # naranja
    (255, 220, 100),  # amarillo
    (100, 255, 150),  # verde
    (100, 200, 255),  # azul
    (200, 150, 255),  # púrpura
]

# Semilla reproducible (opcional)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Mostrar pre-carga visual de clientes desde Excel
PRELOAD_FROM_EXCEL = True
PRELOAD_MAX_VISIBLE = 20

# ---------------------------
# ESTADO COMPARTIDO (entre SimPy y Pygame)
# ---------------------------
lock = threading.Lock()
queue_visual = deque()              # lista de clientes en espera (ids)
servers_visual = [None] * NUM_SERVERS  # qué cliente atiende cada servidor (id or None)

# Después de terminar el servicio, mantener visualmente al cliente unos segundos
# mientras recibe los medicamentos (real-time seconds)
served_visual_hold = {}  # cid -> hold_end_real_time
POST_SERVICE_HOLD_REAL = 2.0
visual_server_release = {}  # server_index -> (cid, hold_end_real_time)

# Optional TTS announcer (pyttsx3). If not available, audio will be skipped.
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

def announce_text(text):
    """Speak the text asynchronously if TTS is available."""
    if not pyttsx3:
        return
    def _run(t):
        try:
            engine = pyttsx3.init()
            engine.say(t)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass
    th = threading.Thread(target=_run, args=(text,), daemon=True)
    th.start()

# Estadísticas
stats = {
    "arrivals": 0,
    "served": 0,
    "total_wait": 0.0,       # acumulado Wq
    "total_system_time": 0.0,
    "server_busy_time": [0.0] * NUM_SERVERS,
    "start_sim": None,
    "end_sim": None
}

# Registro por cliente (para calcular tiempos cuando terminan)
client_records = {}  # client_id -> {"arrive":, "start":, "end":, "server": index}

# Control de ejecución
sim_running = True
sim_finished = False

# ---------------------------
# SIMPY: Procesos y generador
# ---------------------------

def arrival_time(client_idx):
    # Si hay datos reales en Excel, usarlos; sino usar exponencial
    if f"C{client_idx}" in cliente_data:
        return cliente_data[f"C{client_idx}"]["arrival"]
    return random.expovariate(1.0 / MEAN_ARRIVAL)

def service_time(client_idx):
    # Si hay datos reales en Excel, usarlos; sino usar normal
    if f"C{client_idx}" in cliente_data:
        return cliente_data[f"C{client_idx}"]["service"]
    s = random.gauss(MEAN_SERVICE, SD_SERVICE)
    return max(5.0, s)

def cliente_process(env, name, server_resource, server_idx, client_idx):
    """Proceso SimPy para un cliente."""
    global queue_visual, servers_visual, stats, client_records
    arrive = env.now
    # valores especificados en el Excel (si existen)
    arrival_spec = None
    service_spec = None
    if f"C{client_idx}" in cliente_data:
        arrival_spec = cliente_data[f"C{client_idx}"]["arrival"]
        service_spec = cliente_data[f"C{client_idx}"]["service"]

    with lock:
        stats["arrivals"] += 1
        client_records[name] = {
            "arrive": arrive,
            "start": None,
            "end": None,
            "server": None,
            "idx": client_idx,
            "arrival_spec": arrival_spec,
            "service_spec": service_spec,
            "service_duration": None,
        }
        # solo anexar a la cola visual si no fue pre-cargado (evita duplicados)
        if name not in queue_visual:
            queue_visual.append(name)

    # Request a server (we requested a specific server resource chosen by the generator)
    with server_resource.request() as req:
        yield req  # espera en cola a que ese servidor se libere

        assigned_index = server_idx
        with lock:
            # registro inicio de servicio
            client_records[name]["start"] = env.now
            # marcar visualmente el servidor ocupado por este cliente
            servers_visual[assigned_index] = name
            client_records[name]["server"] = assigned_index
            # quitar de la cola visual
            try:
                queue_visual.remove(name)
            except ValueError:
                pass

            # programar hold visual breve (cliente recibe medicamentos visualmente)
            hold_end_start = time.time() + POST_SERVICE_HOLD_REAL
            visual_server_release[assigned_index] = (name, hold_end_start)
            served_visual_hold[name] = hold_end_start
            client_records[name]["visual_release"] = hold_end_start

        # announce audio (non-blocking)
        try:
            announce_text(f"{name} - caja {assigned_index+1}")
        except Exception:
            pass

        # servicio
        st = service_time(client_idx)
        # registrar la duración de servicio asignada (puede venir del Excel o haber sido muestreada)
        with lock:
            client_records[name]["service_duration"] = st
        t0 = env.now
        yield env.timeout(st)
        t1 = env.now

        # terminar servicio: liberar servidor visual y actualizar estadísticas
        with lock:
            client_records[name]["end"] = env.now
            stats["served"] += 1
            wait = client_records[name]["start"] - client_records[name]["arrive"]
            stats["total_wait"] += wait
            system_time = client_records[name]["end"] - client_records[name]["arrive"]
            stats["total_system_time"] += system_time
            # acumular busy time
            if assigned_index is not None:
                stats["server_busy_time"][assigned_index] += (t1 - t0)

def arrival_generator(env, servers):
    """Generador de llegadas. Asigna cada cliente al servidor con menor carga (cola+ocupado)."""
    i = 0
    while env.now < SIM_SECONDS:
        inter = arrival_time(i + 1)
        yield env.timeout(inter)
        i += 1
        cname = f"C{i}"
        # Asignación round-robin para asegurar que todas las cajas atiendan
        chosen_idx = (i - 1) % len(servers)
        env.process(cliente_process(env, cname, servers[chosen_idx], chosen_idx, i))


def sim_thread_fn():
    """Ejecuta SimPy pero pacing en 'tiempo real' basado en TIME_SCALE."""
    global sim_running, sim_finished, stats
    env = simpy.Environment()
    # crear recursos individuales por cajero para balancear la carga
    servers = [simpy.Resource(env, capacity=1) for _ in range(NUM_SERVERS)]

    # start
    stats["start_sim"] = time.time()
    env.process(arrival_generator(env, servers))

    # Ejecutar event-by-event y "dormir" para sincronizar visualmente:
    prev = env.now
    # mientras no llegue a fin, ir dando pasos
    while True:
        if env.peek() is None:
            # no hay próximos eventos -> terminar
            break
        # step 1 evento
        env.step()
        now = env.now
        delta = now - prev
        prev = now
        # dormir en tiempo real proporcional al avance simulacion
        if delta > 0:
            time.sleep(delta / TIME_SCALE)
        if now >= SIM_SECONDS:
            break

    stats["end_sim"] = time.time()
    sim_running = False
    sim_finished = True

# ---------------------------
# PYGAME: Visual y loop
# ---------------------------

def lerp(a, b, t):
    return a + (b - a) * t

def get_client_color(client_idx):
    """Obtener color gradiente basado en índice de cliente."""
    idx = (client_idx - 1) % len(GRADIENT_COLORS)
    return GRADIENT_COLORS[idx]

class VisualClient:
    def __init__(self, cid, idx_in_queue, client_idx):
        self.cid = cid
        self.client_idx = client_idx
        self.color = get_client_color(client_idx)
        self.pos = [QUEUE_X - 50, QUEUE_Y + idx_in_queue * QUEUE_SPACING]
        self.target_pos = self.pos[:]
        self.state = "queue"  # 'queue', 'moving_to_server', 'at_server', 'leaving'
        self.server_index = None
        self.progress = 0.0
        self.glow_intensity = 0.0
        self.preloaded = False
        # motion control (pixels per second)
        self.move_speed = 160.0
        self._last_time = time.time()

    def update_target_for_queue(self, idx):
        self.target_pos = [QUEUE_X, QUEUE_Y + idx * QUEUE_SPACING]

    def move_to_server(self, idx):
        # target inside the booth rectangle (centered)
        booth_w = 320
        booth_h = 70
        sx = SERVER_X + booth_w // 2
        sy = SERVER_Y_START + idx * SERVER_SPACING + booth_h // 2
        self.target_pos = [sx, sy]
        self.state = "moving_to_server"
        self.server_index = idx
        self.progress = 0.0
        self._last_time = time.time()

    def leave_system(self):
        # target off-screen to the right
        self.target_pos = [WINDOW_W + 100, random.randint(10, WINDOW_H-10)]
        self.state = "leaving"
        self.progress = 0.0
        self._last_time = time.time()

    def step(self):
        # movimiento basado en velocidad (pixels/seg)
        now = time.time()
        dt = max(1e-6, now - self._last_time)
        self._last_time = now

        dx = self.target_pos[0] - self.pos[0]
        dy = self.target_pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist > 0.5:
            # mover hacia target con máxima distancia = speed * dt
            max_move = self.move_speed * dt
            move = min(dist, max_move)
            self.pos[0] += (dx / dist) * move
            self.pos[1] += (dy / dist) * move
        else:
            # llegó al target
            self.pos[0], self.pos[1] = self.target_pos[0], self.target_pos[1]
            if self.state == 'moving_to_server':
                self.state = 'at_server'

        # efecto glow
        self.glow_intensity = (math.sin(time.time() * 3) + 1) / 2
        # pequeño balanceo si está en fila para dar vida
        if self.state == 'queue':
            self.pos[1] += math.sin(time.time() * 2 + self.client_idx) * 0.4
        # ligera oscilación cuando está atendido
        if self.state == 'at_server':
            self.pos[1] += math.sin(time.time() * 2 + self.client_idx) * 0.6

def run_pygame():
    global sim_running, sim_finished
    # Esperar hasta que haya al menos un cliente en la fila (o la simulación termine)
    wait_start = time.time()
    while True:
        with lock:
            if queue_visual or stats.get('arrivals', 0) > 0 or sim_finished:
                break
        time.sleep(0.05)
        # si la simulación no genera nunca clientes, evitamos espera infinita
        if time.time() - wait_start > 10 and sim_finished:
            break

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("🏪 Simulación Videojuego - Droguería MENAR 🏪")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont("Arial", 14)
    font_normal = pygame.font.SysFont("Arial", 16)
    font_title = pygame.font.SysFont("Arial", 22, bold=True)
    font_stats = pygame.font.SysFont("Arial", 14, bold=True)

    # Cargar icono de usuario (si existe) y escalarlo al tamaño del cliente
    try:
        base_dir = os.path.dirname(__file__)
        icon_path = os.path.join(base_dir, "icono_usuario.png")
        user_icon_img = pygame.image.load(icon_path).convert_alpha()
        ICON_SIZE = max(16, CLIENT_RADIUS * 2)
        user_icon_img = pygame.transform.smoothscale(user_icon_img, (ICON_SIZE, ICON_SIZE))
    except Exception as e:
        user_icon_img = None
        print(f"⚠ No se pudo cargar 'icono_usuario.png': {e}")

    visual_clients = {}   # cid -> VisualClient
    client_id_counter = 0

    # Pre-cargar visualmente clientes desde Excel (muñequitos en fila)
    if PRELOAD_FROM_EXCEL and cliente_data:
        # solo mostrar hasta PRELOAD_MAX_VISIBLE en la fila
        keys = list(cliente_data.keys())[:PRELOAD_MAX_VISIBLE]
        with lock:
            for idx, cid in enumerate(keys):
                try:
                    client_idx = int(cid[1:])
                except Exception:
                    client_idx = idx + 1
                vc = VisualClient(cid, idx, client_idx)
                vc.preloaded = True
                visual_clients[cid] = vc
                # agregar placeholder a la cola visual si no existe (evita duplicados laterales)
                if cid not in queue_visual:
                    queue_visual.append(cid)

    running = True
    start_time_real = time.time()

    while running:
        # Events Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        # Sync visual state with sim state (incluye clientes en hold post-servicio)
        with lock:
            qlist = list(queue_visual)  # snapshot
            servers_snap = servers_visual[:]
            stats_snap = stats.copy()
            records_snap = client_records.copy()
            served_hold_snap = served_visual_hold.copy()
            visual_server_release_snap = visual_server_release.copy()
            # limpiar expirados (si ya pasó el hold real-time)
            now_rt_lock = time.time()
            for scid, st in list(served_visual_hold.items()):
                if st <= now_rt_lock:
                    del served_visual_hold[scid]
            # liberar servidores visualmente cuando expire su hold
            for sidx, tup in list(visual_server_release.items()):
                vcid, hold_end = tup
                if hold_end <= now_rt_lock:
                    # solo limpiar si el servidor sigue apuntando a ese cliente
                    if servers_visual[sidx] == vcid:
                        servers_visual[sidx] = None
                        # also clear server assignment in client_records so visual loop lets them leave
                        if vcid in client_records:
                            client_records[vcid]["server"] = None
                    del visual_server_release[sidx]

        # Update visual clients: ensure all in queue exist
        # add new clients
        for idx, cid in enumerate(qlist):
            if cid not in visual_clients:
                client_id_counter_local = int(cid[1:])
                visual_clients[cid] = VisualClient(cid, idx, client_id_counter_local)
            # asegurar target pos actualizado (si preloaded, el update_target ajusta posición cuando llegue)
            visual_clients[cid].update_target_for_queue(idx)

        # Asegurar que cualquier cliente que tenga servidor asignado en client_records
        # se mueva hacia ese servidor (evita quedarse en la fila)
        # For any client that has been assigned a server by the sim, ensure it
        # is removed from the visual queue and moves to the booth. Only do this
        # while the client is actually being served (i.e., end is not set).
        with lock:
            for cid, rec in records_snap.items():
                srv_idx = rec.get("server")
                end_t = rec.get("end")
                if srv_idx is not None and end_t is None:
                    # remove from queue_visual if present (visual sync)
                    try:
                        if cid in queue_visual:
                            queue_visual.remove(cid)
                    except Exception:
                        pass
                    # ensure visual client exists
                    if cid not in visual_clients:
                        try:
                            client_id_counter_local = int(cid[1:])
                        except Exception:
                            client_id_counter_local = 0
                        visual_clients[cid] = VisualClient(cid, 0, client_id_counter_local)
                    # mover al servidor si aún no está moviéndose/atendido
                    vc = visual_clients[cid]
                    if vc.server_index != srv_idx or vc.state == 'queue':
                        vc.move_to_server(srv_idx)

        # For servers: if a server has a client and that client is not yet at server -> move it
        for i, sid in enumerate(servers_snap):
            if sid is not None:
                # ensure visual client exists
                if sid not in visual_clients:
                    client_id_local = int(sid[1:])
                    visual_clients[sid] = VisualClient(sid, 0, client_id_local)
                # move client to server
                visual_clients[sid].move_to_server(i)

        # Clients that finished (present in visual but not in any sim structures) -> leave
        now_rt = time.time()
        # Build active served set from client_records (single source of truth) or served_hold
        active_served = set()
        for cid, rec in records_snap.items():
            vrel = rec.get("visual_release")
            if vrel and vrel > now_rt:
                active_served.add(cid)
        # fallback to served_hold_snap if needed
        for cid, t in served_hold_snap.items():
            if t > now_rt:
                active_served.add(cid)

        in_system = set(qlist) | set([s for s in servers_snap if s is not None]) | active_served
        to_remove = []
        for cid, v in visual_clients.items():
            if cid not in in_system:
                if v.state != "leaving":
                    v.leave_system()
            v.step()
            if v.pos[0] > WINDOW_W + 50:
                to_remove.append(cid)

        for cid in to_remove:
            del visual_clients[cid]

        # Drawing
        screen.fill(COLOR_BG)
        
        # --- HEADER AREA ---
        # Título principal
        title = font_title.render("🏪 Simulación Droguería MENAR", True, COLOR_PRIMARY)
        screen.blit(title, (30, 15))
        
        # Subtítulo con tiempos
        real_elapsed = time.time() - start_time_real
        sim_elapsed = (time.time() - stats['start_sim']) * TIME_SCALE if stats['start_sim'] else 0
        sub = font_normal.render(
            f"⏱ Real: {real_elapsed:.1f}s  │  Simulado: {min(SIM_SECONDS, sim_elapsed):.1f}s  │  Escala: {TIME_SCALE:.0f}x",
            True, COLOR_TEXT
        )
        screen.blit(sub, (30, 50))

        # --- QUEUE AREA (IZQUIERDA) ---
        # Caja de fila (adaptable al tamaño de ventana)
        qbox_x = 30
        qbox_y = 60
        qbox_w = 220
        qbox_h = WINDOW_H - qbox_y - 180
        pygame.draw.rect(screen, COLOR_PANEL, (qbox_x, qbox_y, qbox_w, qbox_h), border_radius=12)
        pygame.draw.rect(screen, COLOR_PRIMARY, (qbox_x, qbox_y, qbox_w, qbox_h), 3, border_radius=12)

        qtitle = font_stats.render("📋 FILA DE ESPERA", True, COLOR_PRIMARY)
        screen.blit(qtitle, (qbox_x + 10, qbox_y + 6))

        queue_count = len(qlist)
        qcount_text = font_normal.render(f"En fila: {queue_count}", True, COLOR_ACCENT)
        screen.blit(qcount_text, (qbox_x + 10, qbox_y + 32))

        # --- SERVERS AREA (DERECHA) ---
        # Encabezado de cajeros
        server_title = font_stats.render("💼 CAJEROS", True, COLOR_SUCCESS)
        screen.blit(server_title, (SERVER_X, 105))
        
        for i in range(NUM_SERVERS):
            sx = SERVER_X
            sy = SERVER_Y_START + i * SERVER_SPACING
            
            # booth rectangle con sombra
            pygame.draw.rect(screen, (0, 0, 0, 50), (sx+2, sy+2, 320, 70), border_radius=10)
            pygame.draw.rect(screen, COLOR_PANEL, (sx, sy, 320, 70), border_radius=10)
            
            # borde con color según disponibilidad
            with lock:
                cid = servers_visual[i]
            border_color = COLOR_SUCCESS if not cid else COLOR_ACCENT
            pygame.draw.rect(screen, border_color, (sx, sy, 320, 70), 3, border_radius=10)
            
            # Label "Cajero N"
            cab_label = font_stats.render(f"Cajero #{i+1}", True, COLOR_TEXT)
            screen.blit(cab_label, (sx + 12, sy + 8))
            
            # Estado
            if cid:
                # obtener valores del registro para mostrar tiempos
                rec = records_snap.get(cid, {})
                # preferir el servicio especificado en Excel; si no, mostrar la duración asignada
                svc_spec = rec.get("service_spec") if rec else None
                svc_assigned = rec.get("service_duration") if rec else None
                svc_text_val = svc_spec if svc_spec is not None else svc_assigned
                if svc_text_val is not None:
                    estado = font_normal.render(f"Atendiendo: {cid}  │ S:{int(svc_text_val)}s", True, COLOR_ACCENT)
                else:
                    estado = font_normal.render(f"Atendiendo: {cid}", True, COLOR_ACCENT)
            else:
                estado = font_normal.render("🟢 DISPONIBLE", True, COLOR_SUCCESS)
            screen.blit(estado, (sx + 12, sy + 32))

        # --- VISUAL CLIENTS (icon indicators) ---
            for cid, vc in visual_clients.items():
                # Glow effect (mantener para énfasis visual)
                glow_radius = int(CLIENT_RADIUS * (1 + vc.glow_intensity * 0.3))
                glow_color = tuple(int(c * 0.5) for c in vc.color)
                pygame.draw.circle(screen, glow_color, (int(vc.pos[0]), int(vc.pos[1])), glow_radius, 2)

                # Dibujar icono de usuario centrado en la posición del cliente
                if user_icon_img:
                    icon_rect = user_icon_img.get_rect(center=(int(vc.pos[0]), int(vc.pos[1])))
                    screen.blit(user_icon_img, icon_rect)
                else:
                    # Fallback: dibujo circular original si no hay imagen
                    body_rect = pygame.Rect(0, 0, CLIENT_RADIUS * 2, CLIENT_RADIUS * 2)
                    body_rect.center = (int(vc.pos[0]), int(vc.pos[1]))
                    pygame.draw.ellipse(screen, vc.color, body_rect)
                    pygame.draw.ellipse(screen, (255,255,255), body_rect, 2)

                    # Ojos
                    eye_y = vc.pos[1] - CLIENT_RADIUS * 0.2
                    eye_x_off = CLIENT_RADIUS * 0.4
                    pygame.draw.circle(screen, (255,255,255), (int(vc.pos[0] - eye_x_off), int(eye_y)), 3)
                    pygame.draw.circle(screen, (255,255,255), (int(vc.pos[0] + eye_x_off), int(eye_y)), 3)

                    # Boca simple
                    mouth_y = int(vc.pos[1] + CLIENT_RADIUS * 0.4)
                    pygame.draw.arc(screen, (20,20,20), (vc.pos[0]-6, mouth_y-4, 12, 8), math.pi, 2*math.pi, 2)

                # Etiqueta pequeña debajo del icono
                label = font_small.render(cid, True, (200, 200, 200))
                label_rect = label.get_rect(center=(int(vc.pos[0]), int(vc.pos[1] + CLIENT_RADIUS + 8)))
                screen.blit(label, label_rect)

        # --- PANEL: CLIENTES RECIENTEMENTE ATENDIDOS (AL LADO DE LOS CAJEROS) ---
        # Calculamos un ancho razonable para el panel que quepa a la derecha de los cajeros.
        servers_width = 320
        padding = 10
        desired_w = 300
        recent_x = SERVER_X + servers_width + padding
        # si no cabe a la derecha, colocar justo a la derecha del servidor, ajustando ancho
        if recent_x + desired_w + padding > WINDOW_W:
            recent_w = max(120, WINDOW_W - (SERVER_X + servers_width) - 2 * padding)
            recent_x = SERVER_X + servers_width + padding
        else:
            recent_w = desired_w
        recent_y = SERVER_Y_START
        recent_h = min(220, WINDOW_H - recent_y - 120)

        # Si aún no hay espacio a la derecha, colocar el panel entre la fila y los cajeros
        if recent_w <= 140:
            recent_x = SERVER_X - (recent_w + padding)

        pygame.draw.rect(screen, COLOR_PANEL, (recent_x, recent_y, recent_w, recent_h), border_radius=12)
        pygame.draw.rect(screen, COLOR_PRIMARY, (recent_x, recent_y, recent_w, recent_h), 2, border_radius=12)
        rec_title = font_stats.render("🧾 Clientes atendidos (recientes)", True, COLOR_TEXT)
        screen.blit(rec_title, (recent_x + 8, recent_y + 8))

        # obtener clientes atendidos ordenados por tiempo de fin (desc)
        completed = [(cid, r) for cid, r in records_snap.items() if r.get("end")]
        completed.sort(key=lambda x: x[1].get("end", 0), reverse=True)
        max_show = max(4, int((recent_h - 40) / 22))
        for i, (cid, r) in enumerate(completed[:max_show]):
            ay = recent_y + 36 + i * 22
            arrival_spec = r.get("arrival_spec")
            service_spec = r.get("service_spec")
            service_dur = r.get("service_duration")
            wait = (r.get("start") - r.get("arrive")) if (r.get("start") and r.get("arrive")) else 0.0
            serv_time = (r.get("end") - r.get("start")) if (r.get("end") and r.get("start")) else service_dur
            server_idx = r.get("server")
            srv_label = f"Caj:{server_idx+1}" if server_idx is not None else "Caj: -"
            la = arrival_spec if arrival_spec is not None else int(r.get("arrive",0))
            ls = service_spec if service_spec is not None else (int(service_dur) if service_dur else "?")
            txt = f"{cid}  L:{la}s  S:{ls}s  W:{int(wait)}s  T:{int(serv_time) if serv_time else 0}s  {srv_label}"
            screen.blit(font_small.render(txt, True, COLOR_TEXT), (recent_x + 8, ay))

        # --- STATS PANEL (BOTTOM) ---
        panel_x, panel_y = 12, WINDOW_H - 70
        pygame.draw.rect(screen, COLOR_PANEL, (panel_x, panel_y, WINDOW_W - 24, 58), border_radius=10)
        pygame.draw.rect(screen, COLOR_PRIMARY, (panel_x, panel_y, WINDOW_W - 24, 58), 2, border_radius=10)
        
        # Compute stats
        with lock:
            arrivals = stats["arrivals"]
            served = stats["served"]
            avg_wq = (stats["total_wait"] / served) if served > 0 else 0.0
            avg_w = (stats["total_system_time"] / served) if served > 0 else 0.0
            utilizations = []
            sim_time_elapsed = stats["end_sim"] - stats["start_sim"] if stats["end_sim"] else max(1e-6, time.time() - stats["start_sim"]) if stats["start_sim"] else 0.0
            for b in stats["server_busy_time"]:
                utilizations.append(min(1.0, b / (sim_time_elapsed * TIME_SCALE) if sim_time_elapsed>0 else 0))

        stats_text = (
            f"📊 Llegadas: {arrivals}  │  "
            f"✓ Atendidos: {served}  │  "
            f"⏱ Wq: {avg_wq:.0f}s  │  "
            f"📈 W: {avg_w:.0f}s  │  "
            f"⚡ Util. Prom: {np.mean(utilizations)*100:.0f}%" if utilizations else "0%"
        )
        screen.blit(font_small.render(stats_text, True, COLOR_TEXT), (panel_x + 12, panel_y + 12))

        pygame.display.flip()
        clock.tick(FPS)

        # terminar cuando sim_done
        if sim_finished and not sim_running:
            time.sleep(0.8)
            running = False

    pygame.quit()

# ---------------------------
# PUNTO DE ENTRADA
# ---------------------------

def main():
    # Lanzar hilo SimPy
    t = threading.Thread(target=sim_thread_fn, daemon=True)
    t.start()
    # Ejecutar Pygame en hilo principal
    run_pygame()

    # Al terminar, mostrar resumen en consola
    print("\n--- RESUMEN SIMULACIÓN ---")
    with lock:
        arrivals = stats["arrivals"]
        served = stats["served"]
        avg_wq = (stats["total_wait"] / served) if served > 0 else 0.0
        avg_w = (stats["total_system_time"] / served) if served > 0 else 0.0
        sim_elapsed_real = stats["end_sim"] - stats["start_sim"] if stats["end_sim"] else 0.0
        print(f"Llegadas: {arrivals}")
        print(f"Atendidos: {served}")
        print(f"Wq promedio (sim): {avg_wq:.2f} s")
        print(f"W promedio en sistema (sim): {avg_w:.2f} s")
        print(f"Tiempo real consumido (s): {sim_elapsed_real:.2f}")
        # utilización por cajero
        print("Utilización estimada por cajero (basado en busy_time / sim_time):")
        for i, b in enumerate(stats["server_busy_time"]):
            util = b / SIM_SECONDS if SIM_SECONDS > 0 else 0
            print(f"  Cajero {i+1}: {util*100:.1f}%")
    print("--------------------------\n")

if __name__ == "__main__":
    main()
