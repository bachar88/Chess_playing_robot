import cv2
import numpy as np
from collections import deque
import chess
import chess.engine

# =========================
# ORIENTATION (ta config vraie)
# =========================
ROT_K = 3
FLIP = 0

FILES = "abcdefgh"
RANKS = "87654321"


def rotate_rc(r, c, k):
    if k == 0: return r, c
    if k == 1: return c, 7 - r
    if k == 2: return 7 - r, 7 - c
    if k == 3: return 7 - c, r


def flip_rc(r, c, flip):
    if flip == 0: return r, c
    return r, 7 - c


def map_rc(r, c, rot_k=ROT_K, flip=FLIP):
    r2, c2 = rotate_rc(r, c, rot_k)
    r3, c3 = flip_rc(r2, c2, flip)
    return r3, c3


def sq_to_rc(sq: str):
    sq = sq.lower().strip()
    c = FILES.index(sq[0])
    r = RANKS.index(sq[1])
    return r, c


# =========================
# Grille 9x9 depuis coins internes 7x7
# =========================
def build_full_grid(inner):
    full = np.zeros((9, 9, 2), dtype=np.float32)
    full[1:8, 1:8] = inner

    for r in range(1, 8):
        full[r, 0] = full[r, 1] - (full[r, 2] - full[r, 1])
        full[r, 8] = full[r, 7] + (full[r, 7] - full[r, 6])

    for c in range(9):
        full[0, c] = full[1, c] - (full[2, c] - full[1, c])
        full[8, c] = full[7, c] + (full[7, c] - full[6, c])

    return full


# =========================
# Extraction robuste des cases (warpPerspective)
# =========================
def warp_square(frame, quad_pts, out_size=70):
    quad = np.array(quad_pts, dtype=np.float32)
    dst = np.array([[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(frame, M, (out_size, out_size))


def extract_squares(frame, full_corners, out_size=70, rot_k=ROT_K, flip=FLIP):
    squares = {}
    for r in range(8):
        for c in range(8):
            rr, cc = map_rc(r, c, rot_k, flip)

            tl = full_corners[rr][cc]
            tr = full_corners[rr][cc + 1]
            br = full_corners[rr + 1][cc + 1]
            bl = full_corners[rr + 1][cc]

            name = FILES[c] + RANKS[r]
            squares[name] = warp_square(frame, [tl, tr, br, bl], out_size=out_size)

    return squares


# =========================
# Couleur pièce (avec hystérésis)
# =========================
def piece_color_lab(square_img, prev_color=None):
    h_img, w_img = square_img.shape[:2]
    margin = int(min(h_img, w_img) * 0.25)
    center = square_img[margin:h_img - margin, margin:w_img - margin]


    L, A, B = cv2.split(lab)  # B = jaune/bleu (plus haut = plus jaune)

    # ✅ on ignore :
    # - pixels trop sombres
    # - pixels trop "blancs" (reflets) : L très haut et A/B proches du neutre
    # (ça enlève les highlights qui cassent HSV)
    mask = (L > 60) & (L < 245)

    if int(np.sum(mask)) < 30:
        return prev_color if prev_color else "argent"

    # ✅ score "jaune" robuste : percentile (pas moyenne) pour résister aux reflets
    b_val = B[mask].astype(np.float32)
    b_p70 = float(np.percentile(b_val, 70))
    b_p50 = float(np.percentile(b_val, 50))

    # seuils (à ajuster si besoin)
    # doré => B élevé
    if prev_color == "doree":
        thr = 148  # rester dorée plus facilement
    else:
        thr = 153  # devenir dorée seulement si bien jaune

    return "doree" if (b_p70 > thr and b_p50 > (thr - 5)) else "argent"


def piece_color_hsv(square_img, prev_color=None):
    h_img, w_img = square_img.shape[:2]
    margin = int(min(h_img, w_img) * 0.25)
    center = square_img[margin:h_img - margin, margin:w_img - margin]

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = (s > 35) & (v > 60)

    if int(np.sum(mask)) < 40:
        return prev_color if prev_color else "argent"

    h_mean = float(np.mean(h[mask]))

    b, g, r = cv2.split(center)
    r_mean = float(np.mean(r[mask]))
    b_mean = float(np.mean(b[mask]))
    red_ok = (r_mean > b_mean + 10)

    if prev_color == "doree":
        low, high = 16, 44
    else:
        low, high = 20, 38

    if (low < h_mean < high) and red_ok:
        return "doree"
    return "argent"


# =========================
# Détecteur principal
# =========================
class ChessboardDetector:
    def __init__(self, calib_frames=30):
        self.empty_board_ref = {}
        self.temp = {}
        self.calibration_mode = False
        self.calibration_frames = 0
        self.CALIB_FRAMES = calib_frames

        self.w_var = 0.45
        self.w_grad = 0.35
        self.w_edge = 0.20
        self.OCC_THRESHOLD = 8.5

        self.vote_N = 5
        self.vote_min = 3
        self.occ_history = {}

        self.min_sd = {"mean": 2.0, "std": 1.0, "var": 8.0, "grad": 0.8, "edge": 0.006}

        self.color_N = 7
        self.color_min = 5
        self.color_hist = {}
        self.last_color = {}

    def _metrics(self, square_img):
        h, w = square_img.shape[:2]
        margin = int(min(h, w) * 0.25)
        center = square_img[margin:h - margin, margin:w - margin]

        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        mean = float(np.mean(blur))
        std = float(np.std(blur))
        var = float(np.var(blur))

        sobelx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad = float(np.mean(cv2.magnitude(sobelx, sobely)))

        edges = cv2.Canny(blur, 60, 150)
        edge_density = float(np.mean(edges > 0))

        return {"mean": mean, "std": std, "var": var, "grad": grad, "edge": edge_density}

    def calibrate_empty_board(self, squares):
        if not self.temp:
            for name in squares.keys():
                self.temp[name] = {"mean": [], "std": [], "var": [], "grad": [], "edge": []}

        for name, img in squares.items():
            m = self._metrics(img)
            for k in self.temp[name].keys():
                self.temp[name][k].append(m[k])

        self.calibration_frames += 1

        if self.calibration_frames >= self.CALIB_FRAMES:
            eps = 1e-6
            self.empty_board_ref = {}
            for name in squares.keys():
                ref = {}
                for k, arr in self.temp[name].items():
                    arr = np.array(arr, dtype=np.float32)
                    mu = float(np.mean(arr))
                    sd = float(np.std(arr) + eps)
                    sd = max(sd, self.min_sd.get(k, sd))
                    ref[k + "_mu"] = mu
                    ref[k + "_sd"] = sd
                self.empty_board_ref[name] = ref

            print("✅ Calibration plateau VIDE terminée")
            self.calibration_mode = False
            self.calibration_frames = 0
            self.temp = {}
            return True
        return False

    def stable_occupied(self, square_name, occupied_now):
        if square_name not in self.occ_history:
            self.occ_history[square_name] = deque(maxlen=self.vote_N)
        self.occ_history[square_name].append(1 if occupied_now else 0)
        return sum(self.occ_history[square_name]) >= self.vote_min

    def is_square_occupied(self, square_img, square_name):
        m = self._metrics(square_img)

        if square_name not in self.empty_board_ref:
            score = (m["var"] * 0.35) + (m["grad"] * 0.45) + (m["edge"] * 350.0)
            return score > 45

        ref = self.empty_board_ref[square_name]
        z_var = abs(m["var"] - ref["var_mu"]) / ref["var_sd"]
        z_grad = abs(m["grad"] - ref["grad_mu"]) / ref["grad_sd"]
        z_edge = abs(m["edge"] - ref["edge_mu"]) / ref["edge_sd"]
        z_std = abs(m["std"] - ref["std_mu"]) / ref["std_sd"]

        score = self.w_var * z_var + self.w_grad * z_grad + self.w_edge * z_edge

        near_empty = (z_edge < 2.2 and z_grad < 2.2 and z_var < 2.8 and z_std < 2.5)
        if near_empty:
            return False

        strong_signal = (z_edge > 3.0) or (z_grad > 3.0) or (z_var > 3.2) or (z_std > 3.0)
        occupied = (score > self.OCC_THRESHOLD) and strong_signal

        if occupied and (z_std < 2.0 and z_edge < 2.2 and z_grad < 2.2):
            occupied = False

        return occupied

    def get_last_color(self, square_name):
        return self.last_color.get(square_name, None)

    def stable_color(self, square_name, color_now):
        if square_name not in self.color_hist:
            self.color_hist[square_name] = deque(maxlen=self.color_N)
        self.color_hist[square_name].append(color_now)

        d = sum(1 for c in self.color_hist[square_name] if c == "doree")
        a = len(self.color_hist[square_name]) - d

        if d >= self.color_min:
            stable = "doree"
        elif a >= self.color_min:
            stable = "argent"
        else:
            stable = self.last_color.get(square_name, color_now)

        self.last_color[square_name] = stable
        return stable

    def clear_color(self, square_name):
        if square_name in self.color_hist:
            del self.color_hist[square_name]
        if square_name in self.last_color:
            del self.last_color[square_name]

    def reset_calibration(self):
        self.empty_board_ref = {}
        self.temp = {}
        self.calibration_mode = False
        self.calibration_frames = 0
        self.occ_history = {}
        self.color_hist = {}
        self.last_color = {}


# =========================
# Snapshots + move detection
# =========================
def detect_move(prev_state: dict, new_state: dict):
    became_empty = []
    became_occupied = []
    color_changed_occupied = []

    for sq in prev_state.keys():
        p_occ, p_col = prev_state[sq]
        n_occ, n_col = new_state[sq]

        if p_occ == "O" and n_occ == "E":
            became_empty.append((sq, p_col))
        elif p_occ == "E" and n_occ == "O":
            became_occupied.append((sq, n_col))
        elif p_occ == "O" and n_occ == "O" and p_col != n_col:
            color_changed_occupied.append((sq, p_col, n_col))

    # -------------------------
    # ✅ 1) MOVE normal (2 cases changent)
    # -------------------------
    if len(became_empty) == 1 and len(became_occupied) == 1:
        from_sq, _ = became_empty[0]
        to_sq, col = became_occupied[0]
        return ("move", from_sq, to_sq, col)

    # -------------------------
    # ✅ 2) CAPTURE (1 vide + 1 occupée dont couleur change)
    # -------------------------
    if len(became_empty) == 1 and len(color_changed_occupied) == 1:
        from_sq, _ = became_empty[0]
        to_sq, old_col, new_col = color_changed_occupied[0]
        return ("capture", from_sq, to_sq, new_col, old_col)

    # -------------------------
    # ✅ 3) CASTLING (4 cases changent: roi+tour)
    # became_empty: [king_from, rook_from]
    # became_occupied: [king_to, rook_to]
    # -------------------------
    if len(became_empty) == 2 and len(became_occupied) == 2 and len(color_changed_occupied) == 0:
        empty_sqs = set(sq for sq, _ in became_empty)
        occ_sqs = set(sq for sq, _ in became_occupied)

        # on récupère une "couleur" stable si possible (doree/argent)
        cols = [c for _, c in became_empty if c is not None] + [c for _, c in became_occupied if c is not None]
        col = cols[0] if cols else None

        # patterns UCI du roque (roi only)
        castles = [
            # white
            ("e1", "g1", "h1", "f1"),  # O-O
            ("e1", "c1", "a1", "d1"),  # O-O-O
            # black
            ("e8", "g8", "h8", "f8"),  # O-O
            ("e8", "c8", "a8", "d8"),  # O-O-O
        ]

        for k_from, k_to, r_from, r_to in castles:
            if empty_sqs == {k_from, r_from} and occ_sqs == {k_to, r_to}:
                return ("castle", k_from, k_to, r_from, r_to, col)

    # sinon ambigu
    return ("ambiguous", became_empty, became_occupied, color_changed_occupied)


# =========================
# MAIN
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

pattern_size = (7, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

detector = ChessboardDetector(calib_frames=30)

board_locked = False
inner_corners = None
full_corners = None

snapshots = deque(maxlen=4)
last_move_text = "No move yet"
current_state = None  # état courant (calculé à chaque frame)

print("📸 Camera running")
print("C=calibrer vide | P=snapshot/move | R=reset | Q=quitter")
print(f"✅ Orientation: ROT_K={ROT_K}, FLIP={FLIP}")
board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci(
    r"C:\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"
)
limit = chess.engine.Limit(time=2.0)
human_plays_white = True
if not human_plays_white:
    robot_move = engine.play(board, limit).move
    board.push(robot_move)
    print("🤖 Robot starts:", robot_move.uci())
    print(board, "\n")

last_human_uci = None
awaiting_robot_sync = False
expected_robot_uci = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        new_inner = corners.reshape(7, 7, 2)

        if not board_locked:
            inner_corners = new_inner
            full_corners = build_full_grid(inner_corners)
            board_locked = True
            print("✅ Chessboard detected and locked")
        else:
            dist = np.mean(np.linalg.norm(inner_corners - new_inner, axis=2))
            if dist > 8:
                print("⚠ Board moved → re-locking")
                inner_corners = new_inner
                full_corners = build_full_grid(inner_corners)

    # ===== Analyse + affichage empty/doree/argent =====
    if board_locked and full_corners is not None:
        squares = extract_squares(frame, full_corners, out_size=70, rot_k=ROT_K, flip=FLIP)

        # calibration mode
        if detector.calibration_mode:
            done = detector.calibrate_empty_board(squares)
            cv2.putText(frame, f"CALIBRATION {detector.calibration_frames}/30",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if done:
                cv2.putText(frame, "CALIBRATION DONE!",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # état courant
        current_state = {}

        for name, img in squares.items():
            occ_raw = detector.is_square_occupied(img, name)
            occupied = detector.stable_occupied(name, occ_raw)

            # position texte sur la case (mapping correct)
            r_log, c_log = sq_to_rc(name)
            rr, cc = map_rc(r_log, c_log, ROT_K, FLIP)

            pts = np.array([
                full_corners[rr][cc],
                full_corners[rr][cc + 1],
                full_corners[rr + 1][cc + 1],
                full_corners[rr + 1][cc]
            ], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)

            if occupied:
                prev = detector.get_last_color(name)
                raw_col = piece_color_lab(img, prev_color=prev)
                text = detector.stable_color(name, raw_col)
                color = (0, 255, 0)
                current_state[name] = ("O", text)
            else:
                detector.clear_color(name)
                text = "empty"
                color = (0, 0, 255)
                current_state[name] = ("E", None)

            # ✅ affichage comme avant
            cv2.putText(frame, text, (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # points 9x9
        for rr in range(9):
            for cc in range(9):
                xx, yy = full_corners[rr][cc]
                cv2.circle(frame, (int(xx), int(yy)), 3, (0, 255, 0), -1)

    # ===== UI =====
    cv2.putText(frame, "C: calibrer vide | P: photo/move | R: reset | Q: quit",
                (10, frame.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.putText(frame, f"Last: {last_move_text}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("Chessboard Tracking", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('c'):
        detector.calibration_mode = True
        detector.calibration_frames = 0
        detector.temp = {}
        detector.occ_history = {}
        detector.color_hist = {}
        detector.last_color = {}
        snapshots.clear()
        last_move_text = "No move yet"
        print("🎯 Calibration activée: plateau VIDE ~30 frames (sans bouger)")

    elif key == ord('r'):
        detector.reset_calibration()
        snapshots.clear()
        last_move_text = "No move yet"
        print("🔄 Calibration reset")

    elif key == ord('p'):
        if current_state is None:
            print("⚠ Pas d’état courant (plateau pas détecté).")
            continue
        # ✅ SYNC: après un coup robot, le prochain P sert juste à fixer le nouveau "baseline"
        if awaiting_robot_sync:
            snapshots.clear()
            snapshots.append(dict(current_state))
            awaiting_robot_sync = False
            last_move_text = f"Synced after robot {expected_robot_uci}. Human to play."
            print("✅ Sync OK après le robot:", expected_robot_uci)
            print("➡️ Maintenant joue (humain) puis appuie sur P.")
            continue

        if len(snapshots) == 0:
            snapshots.append(dict(current_state))
            last_move_text = "Snapshot #1 saved"
            print("📷 Snapshot #1 enregistré. Déplace une pièce puis appuie sur P.")
        else:
            prev = snapshots[-1]
            now = dict(current_state)
            snapshots.append(now)

            res = detect_move(prev, now)

            # ----- 1) Si MOVE ou CAPTURE, on construit UCI = fr+to
            if res[0] == "move":
                _, fr, to, col = res
                human_uci = fr + to

            elif res[0] == "capture":
                _, fr, to, new_col, captured_col = res
                human_uci = fr + to

            elif res[0] == "castle":
                _, k_from, k_to, r_from, r_to, col = res
                human_uci = k_from + k_to  # ✅ python-chess veut juste le move du ROI
                print(f"✅ CASTLE detected: King {k_from}->{k_to}  Rook {r_from}->{r_to}")

            else:
                _, be, bo, cc = res
                last_move_text = "Ambiguous"
                print("⚠ mouvement ambigu:")
                print("  became_empty:", be)
                print("  became_occupied:", bo)
                print("  color_changed:", cc)
                continue

            # ----- 2) Anti-doublon (si tu appuies 2 fois)
            if human_uci == last_human_uci:
                print("⏭ même coup détecté, ignoré:", human_uci)
                continue
            # ----- 3) Validation python-chess
            try:
                human_move = chess.Move.from_uci(human_uci)
            except ValueError:
                print(" UCI invalide:", human_uci)
                continue
            if human_move not in board.legal_moves:
                print(" Coup illégal selon python-chess:", human_uci)
                print(board, "\n")
                # ✅ On revient au snapshot AVANT le coup (prev), pas à now
                snapshots.clear()
                snapshots.append(dict(prev))  # <-- IMPORTANT: baseline = position avant
                last_move_text = f"Illegal: {human_uci} (revert to previous position)"
                print("↩️ Revert: remets le plateau comme avant, puis refais un coup légal et appuie sur P.")
                continue
            # ----- 4) Appliquer le coup humain
            board.push(human_move)
            last_human_uci = human_uci
            print("✅ Human plays:", human_uci)
            print(board, "\n")

            # ----- 5) Stockfish répond (si partie pas finie)
            if board.is_game_over():
                print("🏁 Game over! Result:", board.result())
                last_move_text = f"Game over: {board.result()}"
                continue

            robot_move = engine.play(board, limit).move
            board.push(robot_move)

            expected_robot_uci = robot_move.uci()
            awaiting_robot_sync = True

            print("🤖 Robot plays:", expected_robot_uci)
            print(board, "\n")

            last_move_text = f"H:{human_uci} | R:{expected_robot_uci} (press P to sync)"

            # ✅ On ne reset PAS snapshot maintenant (le robot va bouger physiquement)
            # Le prochain appui sur P va juste synchroniser la position après mouvement robot.
            snapshots.clear()
            print("➡️ Appuie sur P une seule fois APRÈS que le robot ait fini de bouger (SYNC).")

cap.release()
engine.quit()
cv2.destroyAllWindows()
