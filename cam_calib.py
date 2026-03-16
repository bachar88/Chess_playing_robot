import cv2
import numpy as np
import time
import chess
import chess.engine
import subprocess

from advanced_test import (analyze_board_state, display_board_state,
                           extract_8x8_squares, extrapolate_corners,
                           draw_chessboard_grid, get_square_color)

# Stockfish engine setup
engine = None
board = None  # python-chess board (tracks the real game)


# ──────────────────────────────────────────────
# 1. STOCKFISH INIT
# ──────────────────────────────────────────────

def test_stockfish_directly(engine_path):
    try:
        proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        output = ""
        for _ in range(20):
            line = proc.stdout.readline()
            output += line
            if "uciok" in line:
                break
        proc.terminate()
        return "uciok" in output
    except Exception as e:
        print(f"Stockfish test failed: {e}")
        return False


def init_engine(engine_path=r"C:\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"):
    global engine, board

    try:
        # Configure le transport pour éviter le crash sur Windows
        import chess.engine
        chess.engine.SimpleEngine.DEFAULT_TIMEOUT = 30  # timeout plus long

        engine = chess.engine.SimpleEngine.popen_uci(
            engine_path,
            setpgrp=False  # important sur Windows
        )
        engine.configure({"Threads": 1, "Hash": 16})  # config légère
        board = chess.Board()
        print("✅ Stockfish engine initialized")

        result = engine.play(board, chess.engine.Limit(time=0.5))
        print(f"✅ Test move: {result.move}")
        return True

    except Exception as e:
        print(f"❌ Failed to initialize Stockfish: {e}")
        return False
# ──────────────────────────────────────────────
# 2. GET BEST MOVE (uses python-chess board directly)
# ──────────────────────────────────────────────

def get_best_move(time_limit=2.0):
    """
    Ask Stockfish for the best move on the current board state.
    Returns UCI string (e.g. 'e2e4') or None.
    """
    global engine, board
    if engine is None or board is None:
        print("❌ Engine not initialized")
        return None
    try:
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move
    except Exception as e:
        print(f"❌ Error getting best move: {e}")
        return None


def apply_stockfish_move(move):
    """
    Apply Stockfish's move to the python-chess board.
    move: chess.Move object
    Returns UCI string for display (e.g. 'e2e4')
    """
    global board
    if move and move in board.legal_moves:
        board.push(move)
        return move.uci()
    else:
        print(f"❌ Illegal move attempted: {move}")
        return None


def apply_human_move(uci_string):
    """
    Apply the human's move to the python-chess board.
    uci_string: e.g. 'e2e4'
    Returns True if valid, False if illegal.
    """
    global board
    try:
        move = chess.Move.from_uci(uci_string)
        if move in board.legal_moves:
            board.push(move)
            return True
        else:
            print(f"❌ Illegal human move: {uci_string}")
            print(f"   Legal moves: {[m.uci() for m in board.legal_moves]}")
            return False
    except Exception as e:
        print(f"❌ Error applying human move: {e}")
        return False


# ──────────────────────────────────────────────
# 3. MOVE DETECTION (fixed null-safety)
# ──────────────────────────────────────────────

def detect_move(previous_state, current_state):
    """
    Detect chess moves by comparing two board states.
    Returns (from_square, to_square, piece_color, captured_color, is_valid)
    """
    if not previous_state or not current_state:
        return None, None, None, None, False

    changed_squares = []
    for square_name in previous_state:
        prev = previous_state[square_name]
        curr = current_state[square_name]
        if prev['has_piece'] != curr['has_piece'] or \
                (prev['has_piece'] and curr['has_piece'] and prev['color'] != curr['color']):
            changed_squares.append(square_name)

    print(f"Changed squares: {changed_squares}")

    if len(changed_squares) == 2:
        from_square, to_square = None, None
        for sq in changed_squares:
            if previous_state[sq]['has_piece'] and not current_state[sq]['has_piece']:
                from_square = sq
            elif not previous_state[sq]['has_piece'] and current_state[sq]['has_piece']:
                to_square = sq

        # Handle capture (color changed on same square)
        if from_square and to_square:
            piece_color = current_state[to_square]['color']
            return from_square, to_square, piece_color, None, True

        # Piece eaten: both squares had pieces but colors swapped
        if from_square is None and to_square is None:
            for sq in changed_squares:
                if previous_state[sq]['color'] != current_state[sq]['color']:
                    to_square = sq
                else:
                    from_square = sq
            if from_square and to_square:
                piece_color = current_state[to_square]['color']
                captured_color = previous_state[to_square]['color']
                return from_square, to_square, piece_color, captured_color, True

    elif len(changed_squares) in (3, 4):
        from_squares, to_squares = [], []
        for sq in changed_squares:
            if previous_state[sq]['has_piece'] and not current_state[sq]['has_piece']:
                from_squares.append(sq)
            elif not previous_state[sq]['has_piece'] and current_state[sq]['has_piece']:
                to_squares.append(sq)

        # Castling: 2 pieces moved, 2 destinations
        if len(from_squares) == 2 and len(to_squares) == 2:
            piece_color = current_state[to_squares[0]]['color']
            return from_squares, to_squares, piece_color, None, True

        # Capture
        if len(to_squares) == 1 and len(from_squares) >= 1:
            to_square = to_squares[0]
            piece_color = current_state[to_square]['color']
            from_square, captured_color = None, None

            for sq in from_squares:
                if previous_state[sq]['color'] == piece_color:
                    from_square = sq
                else:
                    captured_color = previous_state[sq]['color']

            if not captured_color and previous_state[to_square]['has_piece']:
                captured_color = previous_state[to_square]['color']

            if from_square:
                return from_square, to_square, piece_color, captured_color, True

    return None, None, None, None, False


def get_human_move_from_cv(move_start_state, current_display_state):
    """Returns UCI string or None if detection failed."""
    from_square, to_square, piece_color, captured_color, is_valid = detect_move(
        move_start_state, current_display_state
    )
    if not is_valid or from_square is None or to_square is None:
        return None
    if isinstance(from_square, list):  # Castling
        return None  # Handle separately if needed
    return from_square + to_square


# ──────────────────────────────────────────────
# 4. MAIN LOOP
# ──────────────────────────────────────────────

squares_init = [f"{file}{rank}" for file in "hgfedcba" for rank in range(1, 9)]
maximum_contours = {square: 0 for square in squares_init}
calibrated = False

def main():
    global maximum_contours, calibrated, squares

    if not init_engine(r"C:\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"):
        print("❌ Cannot start without Stockfish. Exiting.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pattern_size = (7, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    board_locked = False
    game_started = False
    inner_corners = None
    full_corners = None

    last_analysis_time = 0
    analysis_interval = 1.0
    board_state = {}
    current_display_state = {}
    pending_sf_move = None
    move_history = []
    captured_pieces = []
    waiting_for_move = False
    move_start_state = None
    last_move_text = ""
    last_move_time = 0
    move_display_duration = 5.0

    current_turn = "white (silver)"
    turn_text = "Silver's turn"

    print("📸 Camera running...")
    print("Press 'b' to lock board | 's' to start | 'm' after move | 'r' reset | 'q' quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('hsv',hsv)
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_FAST_CHECK
        )

        key = cv2.waitKey(1) & 0xFF

        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            new_inner = corners.reshape(7, 7, 2)

            if not board_locked:
                inner_corners = new_inner
                full_corners = extrapolate_corners(inner_corners)
                squares, squares_info = extract_8x8_squares(frame, full_corners)
                board_state = analyze_board_state(squares, squares_info, maximum_contours)
                current_display_state = board_state.copy()
                display_board_state(board_state)
                display_frame = draw_chessboard_grid(display_frame, full_corners, current_display_state)

                cv2.putText(display_frame, "Calibrating...", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if key == ord('b'):
                    board_locked = True
                    print("🔒 Board locked! Place pieces, then press 's'.")

        if board_locked and full_corners is not None:
            squares, squares_info = extract_8x8_squares(frame, full_corners)

            if not calibrated:
                for _ in range(100):
                    squares, squares_info = extract_8x8_squares(frame, full_corners)
                    for square_name, square_img in squares.items():
                        g = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
                        blur = cv2.GaussianBlur(g, (9, 9), 0)
                        edges = cv2.Canny(blur, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size
                        if maximum_contours[square_name] < edge_density:
                            maximum_contours[square_name] = edge_density
                calibrated = True

            current_time = time.time()
            if current_time - last_analysis_time > analysis_interval:
                latest_board_state = analyze_board_state(squares, squares_info, maximum_contours)
                if not game_started:
                    board_state = latest_board_state.copy()
                    current_display_state = board_state.copy()
                elif waiting_for_move:
                    current_display_state = latest_board_state.copy()
                last_analysis_time = current_time

            display_frame = draw_chessboard_grid(display_frame, full_corners, current_display_state)

            y_offset = 160
            if not game_started:
                cv2.putText(display_frame, "SETUP - Place pieces then press 's'",
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            else:
                turn_color = (255, 255, 255) if current_turn == "white (silver)" else (255, 215, 0)
                cv2.putText(display_frame, turn_text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, turn_color, 2)
                y_offset += 30
                cv2.putText(display_frame, "Make move then press 'm'",
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if last_move_text and (current_time - last_move_time) < move_display_duration:
                    cv2.putText(display_frame, last_move_text, (20, y_offset + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            silver_count = sum(1 for info in current_display_state.values()
                               if info['has_piece'] and info['color'] == 'white (silver)')
            gold_count = sum(1 for info in current_display_state.values()
                             if info['has_piece'] and info['color'] == 'blue (gold)')
            cv2.putText(display_frame, f"Silver: {silver_count} | Gold: {gold_count} | Moves: {len(move_history)}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            status = "BOARD LOCKED" + (" - GAME ACTIVE" if game_started else " - SETUP")
            color = (0, 255, 0) if game_started else (255, 165, 0)
        else:
            status = "CALIBRATING - Show chessboard"
            color = (0, 165, 255)

        cv2.putText(display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Chessboard Detection", display_frame)

        # ── KEY HANDLING ──
        if key == ord('q'):
            break

        elif key == ord('r'):
            pending_sf_move = None
            board_locked = False
            game_started = False
            inner_corners = None
            full_corners = None
            board_state = {}
            current_display_state = {}
            move_history = []
            captured_pieces = []
            waiting_for_move = False
            move_start_state = None
            current_turn = "white (silver)"
            turn_text = "Silver's turn"
            last_move_text = ""
            calibrated = False
            maximum_contours = {sq: 0 for sq in squares_init}
            if board:
                board.reset()
            print("🔄 Full reset.")
        elif key == ord('g') and board_locked:
            # Reset game state only — keep board locked, corners, and calibration
            game_started = False
            pending_sf_move = None
            waiting_for_move = False
            move_start_state = None
            move_history = []
            captured_pieces = []
            current_turn = "white (silver)"
            turn_text = "Silver's turn"
            last_move_text = ""
            last_move_time = 0
            board_state = current_display_state.copy()

            if board:
                board.reset()

            print("🔄 Game restarted! Board locked, calibration kept. Press 's' to start.")
        elif key == ord('s') and board_locked and not game_started:


            game_started = True
            waiting_for_move = True
            board_state = current_display_state.copy()
            move_start_state = board_state.copy()
            captured_pieces = []
            if board:
                board.reset()  # Reset python-chess to standard start
            print("🎮 Game started! Silver's turn.")


        elif key == ord('m') and board_locked and game_started and waiting_for_move:

            print("\n--- Checking for move ---")

            if current_turn == "white (silver)":

                # ── HUMAN MOVE ──

                uci_move = get_human_move_from_cv(move_start_state, current_display_state)

                if uci_move is None:

                    print("❌ No valid move detected. Try again.")

                else:

                    if apply_human_move(uci_move):

                        from_sq = uci_move[:2]

                        to_sq = uci_move[2:]

                        last_move_text = f"Silver: {from_sq} -> {to_sq}"

                        last_move_time = time.time()

                        move_history.append({'uci': uci_move, 'player': current_turn})

                        print(f"✅ Human move applied: {uci_move}")

                        # Snapshot board AFTER human move, before Stockfish physically moves

                        board_state = current_display_state.copy()

                        move_start_state = board_state.copy()

                        # Switch turn to Stockfish — user must press 'm' again after moving pieces

                        current_turn = "blue (gold)"

                        turn_text = "Stockfish's turn — compute move then press 'm'"

                        # Stockfish computes best move now and stores it

                        sf_move = get_best_move(time_limit=2.0)

                        if sf_move:

                            print(
                                f"🤖 Stockfish suggests: {sf_move.uci()} — make that move on the board, then press 'm'")

                            last_move_text = f"Stockfish move to make: {sf_move.uci()}"

                            last_move_time = time.time()

                            # Store pending move so next 'm' press can apply it

                            pending_sf_move = sf_move

                        else:

                            print("❌ Stockfish couldn't find a move (game over?)")

                            pending_sf_move = None

                    else:

                        print(f"❌ Illegal move '{uci_move}'")


            elif current_turn == "blue (gold)":

                # ── STOCKFISH MOVE CONFIRMATION ──

                # Player physically moved the piece on the board, now confirm it

                if pending_sf_move:

                    applied = apply_stockfish_move(pending_sf_move)

                    if applied:

                        print(f"✅ Stockfish move confirmed: {applied}")

                        last_move_text = f"Stockfish played: {applied}"

                        last_move_time = time.time()

                        move_history.append({'uci': applied, 'player': 'stockfish'})

                    else:

                        print("❌ Could not apply Stockfish move")

                else:

                    print("❌ No pending Stockfish move found")

                board_state = current_display_state.copy()

                move_start_state = board_state.copy()

                current_turn = "white (silver)"

                turn_text = "Silver's turn"

                pending_sf_move = None
    cap.release()
    cv2.destroyAllWindows()
    if engine:
        engine.quit()

    print(f"\n✅ Done | Moves: {len(move_history)}")


if __name__ == "__main__":
    main()