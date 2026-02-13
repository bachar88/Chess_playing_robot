import cv2
import numpy as np
import time
from collections import defaultdict


# =========================
# Utility functions
# =========================

def extrapolate_corners(inner):
    """Extrapolate 7x7 corners to 9x9 board corners"""
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


def board_moved(old, new, threshold=15.0):  # Increased threshold
    """Check if board has moved significantly"""
    if old is None or new is None:
        return True
    dist = np.mean(np.linalg.norm(old - new, axis=2))
    return dist > threshold


def extract_8x8_squares(frame, full_corners):
    """Extract individual chess squares from the board"""
    squares = {}
    squares_info = {}  # Store square coordinates for piece detection
    files = "abcdefgh"
    ranks = "87654321"  # top of image = rank 8

    for r in range(8):
        for c in range(8):
            pts = np.array([
                full_corners[r][c],
                full_corners[r][c + 1],
                full_corners[r + 1][c + 1],
                full_corners[r + 1][c]
            ], dtype=np.int32)

            x, y, w, h = cv2.boundingRect(pts)

            # Ensure we don't go out of bounds
            y_end = min(y + h, frame.shape[0])
            x_end = min(x + w, frame.shape[1])

            if y < y_end and x < x_end:
                square = frame[y:y_end, x:x_end]
                square_name = files[c] + ranks[r]
                squares[square_name] = square
                squares_info[square_name] = {
                    'coords': (x, y, x_end - x, y_end - y),
                    'center': (x + (x_end - x) // 2, y + (y_end - y) // 2),
                    'pts': pts
                }

    return squares, squares_info


def detect_piece_in_square(square_img, square_name):
    """
    Detect if a piece is present in a square and determine its color
    Returns: (has_piece, piece_color, confidence)
    """
    if square_img is None or square_img.size == 0:
        return False, None, 0.0

    # Get square dimensions
    h, w = square_img.shape[:2]

    # Focus on center region (avoid edges)
    margin = int(min(h, w) * 0.2)
    center_region = square_img[margin:h - margin, margin:w - margin]

    if center_region.size == 0:
        return False, None, 0.0

    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)

    # Calculate average color in center region
    avg_color = np.mean(center_region, axis=(0, 1))
    avg_hsv = np.mean(hsv, axis=(0, 1))

    # Get square color (light or dark)
    square_color = get_square_color(square_img, square_name)

    # Method 1: Check for piece by analyzing edges
    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5), 0 )
    # Apply edge detection
    edges = cv2.Canny(blur, 50, 150)
    #cv2.imshow(f"edges de {square_name}: ",edges)
    # Calculate edge density (pieces have more edges)
    contours, hierarchies = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    edge_density = np.sum(edges > 0) / edges.size
    if square_name == "e4" :
        print("Edges is equal : ", edge_density)
        print("Contours = ", len(contours))
        cv2.imshow("e4 edges : ", edges)
    # Method 2: Check for color contrast with square
    # Empty squares have more uniform color
    std_color = np.std(square_img, axis=(0, 1))
    color_variation = np.mean(std_color)

    # Method 3: Check brightness distribution
    brightness = np.mean(gray)
    brightness_std = np.std(gray)

    # Combined piece detection logic
    has_piece = False
    confidence = 0.0

    # Edge-based detection
    if len(contours) > 4:  # Piece has more edges
        has_piece = True
        confidence += 0.4

    # Color variation detection
    if color_variation > 15:  # Piece has more color variation
        #has_piece = True
        confidence += 0.3

    # Brightness variation detection
    if brightness_std > 20:  # Piece has more brightness variation
        #has_piece = True
        confidence += 0.3

    confidence = min(confidence, 1.0)

    if not has_piece:
        return False, None, 0.0

    # Determine piece color (white or black)
    piece_color = detect_piece_color(center_region, square_color)

    return True, piece_color, confidence


def get_square_color(square_img, square_name):
    """Determine if square is light or dark based on chessboard pattern"""
    # Chessboard pattern: a1 is dark, h8 is light
    files = "abcdefgh"
    ranks = "87654321"

    file_idx = files.index(square_name[0])
    rank_idx = ranks.index(square_name[1])

    # Chessboard pattern formula: (file + rank) % 2 == 0 means dark
    is_dark = (file_idx + rank_idx) % 2 == 0

    return 'dark' if is_dark else 'light'


def detect_piece_color(square_region, square_color):
    """
    Detect if piece is white or black
    Returns: 'white', 'black', or 'unknown'
    """
    if square_region.size == 0:
        return 'unknown'

    # Convert to grayscale for brightness analysis
    gray = cv2.cvtColor(square_region, cv2.COLOR_BGR2GRAY)

    # Calculate average brightness
    avg_brightness = np.mean(gray)

    # Calculate brightness standard deviation
    brightness_std = np.std(gray)

    # Calculate color histogram
    hsv = cv2.cvtColor(square_region, cv2.COLOR_BGR2HSV)

    # Analyze saturation (white pieces have lower saturation)
    avg_saturation = np.mean(hsv[:, :, 1])

    # Analyze value (brightness in HSV)
    avg_value = np.mean(hsv[:, :, 2])

    # Analyze color channels separately
    b, g, r = cv2.split(square_region)
    avg_r = np.mean(r)
    avg_g = np.mean(g)
    avg_b = np.mean(b)

    # White piece characteristics:
    # - High brightness
    # - Low saturation
    # - Balanced RGB (not too much of any color)
    is_white = (
            avg_brightness > 100 and  # Bright
            avg_saturation < 90 and  # Desaturated
            abs(avg_r - avg_g) < 25 and  # Balanced colors
            abs(avg_g - avg_b) < 35 and
            brightness_std > 15  # Some variation (not uniform)
    )

    # Black piece characteristics:
    # - Low brightness
    # - Higher saturation (dark but colorful)
    # - Often blue/red tint for dark pieces
    is_black = (
            avg_brightness < 80 and  # Dark
            avg_saturation >120 and  # Saturated
            abs(avg_r - avg_g) > 10 and  # Balanced colors
            abs(avg_g - avg_b) > 20
            #brightness_std > 20 #and  # More variation
            #(avg_saturation > 120 or  # Either saturated color
            # avg_value < 70)  # Or very dark
    )

    if is_white and not is_black:
        return 'white'
    elif is_black and not is_white:
        return 'black'

    # Fallback logic based on brightness
    if avg_brightness > 100:
        return 'white'
    elif avg_brightness < 80:
        return 'black'
    else:
        return 'unknown'


def draw_chessboard_grid(frame, full_corners, board_state=None):
    """Draw the chessboard grid with piece detection info"""
    if full_corners is None:
        return frame

    # Draw lines
    for r in range(9):
        for c in range(9):
            if r < 8:
                # Vertical lines
                pt1 = tuple(full_corners[r, c].astype(int))
                pt2 = tuple(full_corners[r + 1, c].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            if c < 8:
                # Horizontal lines
                pt1 = tuple(full_corners[r, c].astype(int))
                pt2 = tuple(full_corners[r, c + 1].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw corners
    for r in range(9):
        for c in range(9):
            center = tuple(full_corners[r, c].astype(int))
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

    # Draw piece detection info if available
    if board_state:
        for square_name, info in board_state.items():
            if info['has_piece']:
                # Get square center from corners (simplified)
                file_idx = ord(square_name[0]) - ord('a')
                rank_idx = 8 - int(square_name[1])  # Convert rank to 0-7

                if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
                    # Calculate center between four corners
                    x1 = full_corners[rank_idx, file_idx, 0]
                    y1 = full_corners[rank_idx, file_idx, 1]
                    x2 = full_corners[rank_idx + 1, file_idx + 1, 0]
                    y2 = full_corners[rank_idx + 1, file_idx + 1, 1]

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Draw circle for piece
                    color = (255, 255, 255) if info['color'] == 'white' else (0, 0, 0)
                    thickness = -1 if info['color'] == 'white' else 2
                    cv2.circle(frame, (center_x, center_y), 15, color, thickness)

                    # Add confidence text
                    cv2.putText(frame, f"{info['confidence']:.1f}",
                                (center_x - 10, center_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    return frame


def analyze_board_state(squares, squares_info):
    """Analyze all squares to detect pieces"""
    board_state = {}

    for square_name, square_img in squares.items():
        if square_name in squares_info and square_img is not None and square_img.size > 0:
            has_piece, piece_color, confidence = detect_piece_in_square(square_img, square_name)

            board_state[square_name] = {
                'has_piece': has_piece,
                'color': piece_color,
                'confidence': confidence,
                'coords': squares_info[square_name]['coords']
            }
        else:
            board_state[square_name] = {
                'has_piece': False,
                'color': None,
                'confidence': 0.0,
                'coords': (0, 0, 0, 0)
            }

    return board_state


def display_board_state(board_state):
    """Display board state in console"""
    print("\n" + "=" * 50)
    print("CHESSBOARD STATE ANALYSIS")
    print("=" * 50)

    files = "abcdefgh"
    ranks = "87654321"

    # Create board representation
    board_grid = [["  " for _ in range(8)] for _ in range(8)]

    for square_name, info in board_state.items():
        file_idx = files.index(square_name[0])
        rank_idx = ranks.index(square_name[1])

        if info['has_piece']:
            piece_char = 'W' if info['color'] == 'white' else 'B'
            conf_char = str(int(info['confidence'] * 10))  # 0-9
            board_grid[rank_idx][file_idx] = f"{piece_char}{conf_char}"
        else:
            board_grid[rank_idx][file_idx] = "--"

    # Print board
    print("  " + " ".join(files.upper()))
    for i in range(8):
        print(f"{8 - i} ", end="")
        for j in range(8):
            print(board_grid[i][j], end=" ")
        print(f" {8 - i}")
    print("  " + " ".join(files.upper()))

    # Print statistics
    white_pieces = sum(1 for info in board_state.values()
                       if info['has_piece'] and info['color'] == 'white')
    black_pieces = sum(1 for info in board_state.values()
                       if info['has_piece'] and info['color'] == 'black')

    print(f"\n📊 Statistics:")
    print(f"White pieces detected: {white_pieces}")
    print(f"Black pieces detected: {black_pieces}")
    print(f"Total pieces: {white_pieces + black_pieces}")
    print("=" * 50)


# =========================
# Main program
# =========================
def main():
    # Camera setup
    cap = cv2.VideoCapture(1)  # USB webcam

    # Camera settings for better detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    pattern_size = (7, 7)  # Inner corners of chessboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # State variables
    board_locked = False
    inner_corners = None
    full_corners = None
    last_detection_time = time.time()
    last_analysis_time = 0
    analysis_interval = 2.0  # Analyze board every 2 seconds

    # Store board state
    board_state = {}

    print("=" * 60)
    print("CHESSBOARD PIECE DETECTION PROGRAM")
    print("=" * 60)
    print("Instructions:")
    print("1. Show a chessboard with pieces to the camera")
    print("2. Ensure good lighting and clear view")
    print("3. Chessboard will be auto-detected")
    print("4. Pieces will be detected with colors")
    print("5. Press 'q' to quit")
    print("6. Press 'r' to reset detection")
    print("7. Press 'a' to analyze board now")
    print("=" * 60)

    print("\n📸 Camera running...")
    print("🟢 Waiting for chessboard...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠ Failed to grab frame")
            time.sleep(0.1)
            continue

        # Create a copy for drawing
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray : ", gray)
        cv2.imshow("edged", cv2.Canny(cv2.GaussianBlur(gray,(5,5), 0), 50, 150))
        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FILTER_QUADS +
            cv2.CALIB_CB_FAST_CHECK
        )
    while True:
        if found:
            # Refine corners
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            new_inner = corners.reshape(7, 7, 2)

            if not board_locked:
                # First detection - lock the board
                inner_corners = new_inner
                full_corners = extrapolate_corners(inner_corners)
                board_locked = True
                last_detection_time = time.time()
                print("✅ Chessboard detected and locked!")
            else:
                # Check if board moved
                if board_moved(inner_corners, new_inner):
                    print("⚠ Board moved - re-locking...")
                    inner_corners = new_inner
                    full_corners = extrapolate_corners(inner_corners)
                    last_detection_time = time.time()

        # If board is locked, extract and analyze squares
        if board_locked and full_corners is not None:
            # Extract squares
            squares, squares_info = extract_8x8_squares(frame, full_corners)

            # Analyze board state periodically
            current_time = time.time()
            if current_time - last_analysis_time > analysis_interval:
                board_state = analyze_board_state(squares, squares_info)
                display_board_state(board_state)
                last_analysis_time = current_time


            # Draw grid with piece detection
            display_frame = draw_chessboard_grid(display_frame, full_corners, board_state)

            # Show sample squares with piece detection info
            for square_name in ["e4", "d5", "a1", "h8"]:
                if square_name in squares and squares[square_name].size > 0:
                    square_display = cv2.resize(squares[square_name], (200, 200))

                    # Add piece detection info to square display
                    if square_name in board_state and board_state[square_name]['has_piece']:
                        info = board_state[square_name]
                        color_text = f"Piece: {info['color'].upper()}"
                        conf_text = f"Conf: {info['confidence']:.2f}"

                        # Draw background for text
                        cv2.rectangle(square_display, (0, 0), (200, 40), (0, 0, 0), -1)

                        # Add text
                        cv2.putText(square_display, color_text, (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 1)
                        cv2.putText(square_display, conf_text, (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 0), 1)

                    cv2.imshow(f"Square {square_name}", square_display)
            #for square_name in squares :
            #    square_display = cv2.resize(squares[square_name], (200, 200))
            #    blur = cv2.GaussianBlur(cv2.cvtColor(square_display, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            #    square_canny= cv2.Canny(blur, 50, 150)
            #    cv2.imshow(f"Square {square_name}", square_canny)

            # Add board statistics overlay
            white_count = sum(1 for info in board_state.values()
                              if info['has_piece'] and info['color'] == 'white')
            black_count = sum(1 for info in board_state.values()
                              if info['has_piece'] and info['color'] == 'black')

            stats_text = f"White: {white_count} | Black: {black_count} | Total: {white_count + black_count}"
            cv2.putText(display_frame, stats_text, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            status = "BOARD LOCKED - PIECES DETECTED"
            color = (0, 255, 0)
        else:
            status = "SEARCHING FOR BOARD"
            color = (0, 165, 255)

        # Display status and instructions
        cv2.putText(display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        instruction_text = "'q':quit, 'r':reset, 'a':analyze now"
        cv2.putText(display_frame, instruction_text,
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add FPS counter
        fps_text = f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}"
        cv2.putText(display_frame, fps_text, (display_frame.shape[1] - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show main window
        cv2.imshow("Chessboard Piece Detection", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            board_locked = False
            inner_corners = None
            full_corners = None
            board_state = {}
            # Close square windows
            for square_name in ["e4", "d5", "a1", "h8"]:
                cv2.destroyWindow(f"Square {square_name}")
            print("\n🔄 Detection reset - show chessboard again")
        elif key == ord('a') and board_locked:
            # Analyze board immediately
            squares, squares_info = extract_8x8_squares(frame, full_corners)
            board_state = analyze_board_state(squares, squares_info)
            display_board_state(board_state)
            last_analysis_time = time.time()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Program terminated")


if __name__ == "__main__":
    main()