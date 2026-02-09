import cv2
import numpy as np

# =========================
# Utility functions
# =========================

def extrapolate_corners(inner):
    full = np.zeros((9, 9, 2), dtype=np.float32)
    full[1:8, 1:8] = inner

    # Horizontal extension
    for r in range(1, 8):
        full[r, 0] = full[r, 1] - (full[r, 2] - full[r, 1])
        full[r, 8] = full[r, 7] + (full[r, 7] - full[r, 6])

    # Vertical extension
    for c in range(9):
        full[0, c] = full[1, c] - (full[2, c] - full[1, c])
        full[8, c] = full[7, c] + (full[7, c] - full[6, c])

    return full


def board_moved(old, new, threshold=8):
    dist = np.mean(np.linalg.norm(old - new, axis=2))
    return dist > threshold


def extract_8x8_squares(frame, full_corners):
    squares = {}
    files = "abcdefgh"
    ranks = "87654321"   # top of image = rank 8

    for r in range(8):
        for c in range(8):
            pts = np.array([
                full_corners[r][c],
                full_corners[r][c+1],
                full_corners[r+1][c+1],
                full_corners[r+1][c]
            ], dtype=np.int32)

            x, y, w, h = cv2.boundingRect(pts)
            squares[files[c] + ranks[r]] = frame[y:y+h, x:x+w]

    return squares


# =========================
# Webcam + Chessboard logic
# =========================
phone_ip="192.168.0.90"
wifi_url = f'http://{phone_ip}:4747/video' 
cap = cv2.VideoCapture(wifi_url) #he4i li lezmek tbadalha akel url_wifi
#cap = cv2.VideoCapture(0)
pattern_size = (7, 7)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

board_locked = False
inner_corners = None
full_corners = None

print("📸 Camera running")
print("🟢 Waiting for chessboard...")
print("Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        new_inner = corners.reshape(7, 7, 2)

        if not board_locked:
            inner_corners = new_inner
            full_corners = extrapolate_corners(inner_corners)
            board_locked = True
            print("✅ Chessboard detected and locked")

        else:
            if board_moved(inner_corners, new_inner):
                print("⚠ Board moved → re-locking")
                inner_corners = new_inner
                full_corners = extrapolate_corners(inner_corners)

    # If board is locked, extract squares
    if board_locked:
        squares = extract_8x8_squares(frame, full_corners)

        # Draw grid (debug)
        for r in range(9):
            for c in range(9):
                x, y = full_corners[r][c]
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Show one square for testing
        cv2.imshow("Square e4", squares["e4"])

    cv2.imshow("Chessboard Tracking", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
