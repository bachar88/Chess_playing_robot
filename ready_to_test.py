import time
import cv2
import chess
import chess.engine
from advanced_test import analyze_board_state, display_board_state, extract_8x8_squares, extrapolate_corners, \
    draw_chessboard_grid



def detect_move(previous_state, current_state):
    """
    Detect chess moves by comparing two board states.
    Returns a tuple (from_square, to_square, piece_color, captured_color, is_valid)
    """
    if not previous_state or not current_state:
        return None, None, None, None, False

    changed_squares = []

    # Find squares that changed
    for square_name in previous_state.keys():
        prev_piece = previous_state[square_name]
        curr_piece = current_state[square_name]

        # Check if piece presence or color changed
        if prev_piece['has_piece'] != curr_piece['has_piece'] or \
                (prev_piece['has_piece'] and curr_piece['has_piece'] and
                 prev_piece['color'] != curr_piece['color']):
            changed_squares.append(square_name)

    print(f"Changed squares: {changed_squares}")  # Debug

    # Analyze changes to detect move
    if len(changed_squares) == 2:
        # Typical move: one square loses a piece, another gains one
        from_square = None
        to_square = None

        for square in changed_squares:
            if previous_state[square]['has_piece'] and not current_state[square]['has_piece']:
                from_square = square
            elif not previous_state[square]['has_piece'] and current_state[square]['has_piece']:
                to_square = square



        if changed_squares[0] == from_square :
            count=changed_squares[1]
        else :
            count = changed_squares[0]
        if from_square and to_square:
            piece_color = current_state[to_square]['color']
            # No capture in normal move
            return from_square, to_square, piece_color, None, True
        elif from_square and previous_state[count]['color'] != current_state[count]['color']:
            piece_color = previous_state[from_square]['color']
            print(f"{from_square} -> {count} -> {piece_color} here a piece was eaten")
            return from_square, count, piece_color, None, True

    elif len(changed_squares) == 3 or len(changed_squares) == 4:
        # Capture: piece moves to square with opponent's piece
        from_squares = []
        to_squares = []
        removed_squares = []  # Squares that lost pieces
        for square in changed_squares:
            if previous_state[square]['has_piece'] and not current_state[square]['has_piece']:
                from_squares.append(square)
            elif not previous_state[square]['has_piece'] and current_state[square]['has_piece']:
                to_squares.append(square)
        if len(to_squares)==2 and len(from_squares)==2 :
            piece_color = current_state[to_squares[0]]['color']
            # No capture in normal move
            return from_squares, to_squares, piece_color, None, True

        for square in changed_squares:
            if previous_state[square]['has_piece'] and not current_state[square]['has_piece']:
                removed_squares.append(square)
                if previous_state[square]['color']:  # This was a piece that moved or was captured
                    from_squares.append(square)
            elif not previous_state[square]['has_piece'] and current_state[square]['has_piece']:
                to_squares.append(square)

        print(f"From squares: {from_squares}, To squares: {to_squares}")  # Debug

        if len(from_squares) >= 1 and len(to_squares) == 1:
            # One destination square found
            to_square = to_squares[0]

            # Find which from square has the same color as the piece in to_square
            piece_color = current_state[to_square]['color']
            from_square = None
            captured_color = None

            for square in from_squares:
                if previous_state[square]['has_piece'] and previous_state[square]['color'] == piece_color:
                    from_square = square
                elif previous_state[square]['has_piece'] and previous_state[square]['color'] != piece_color:
                    captured_color = previous_state[square]['color']

            if from_square:
                # If we found a captured color but no explicit captured square,
                # the captured piece was on the destination square
                if not captured_color and previous_state[to_square]['has_piece']:
                    captured_color = previous_state[to_square]['color']

                return from_square, to_square, piece_color, captured_color, True

    return None, None, None, None, False

def get_human_move_from_cv(move_start_state,current_display_state):
    from_square, to_square, piece_color, captured_color, is_valid = detect_move(move_start_state,current_display_state)
    return from_square+to_square

def chess_engine():
    time.sleep(2)
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    limit = chess.engine.Limit(time=2.0)
    human_plays_white = True
    if not human_plays_white:
        robot_result = engine.play(board, limit)
        robot_move = robot_result.move

        board.push(robot_move)

    last_move = None

    while not board.is_game_over():
        human_move_uci = get_human_move_from_cv()
        if human_move_uci is None:
            continue

        if human_move_uci == last_move:
            continue

        try:
            human_move = chess.Move.from_uci(human_move_uci)
        except ValueError:
            print("Invalid UCI:", human_move_uci)
            continue

        if human_move not in board.legal_moves:
            print("Illegal move:", human_move_uci)
            continue

        last_move = human_move_uci
        board.push(human_move)
        print("Human plays:", human_move.uci())
        print(board, "\n")

        if board.is_game_over():
            break

        robot_result = engine.play(board, limit)
        robot_move = robot_result.move

        board.push(robot_move)
        print("Robot plays:", robot_move.uci())
        print(board, "\n")

    print("Game over!")
    print("Result:", board.result())

    engine.quit()
maximum_contours = 0
def main():
    # Camera setup
    global maximum_contours
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    pattern_size = (7, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # State variables
    board_locked = False
    game_started = False
    inner_corners = None
    full_corners = None

    last_analysis_time = 0
    analysis_interval = 1.0
    board_state = {}
    current_display_state = {}  # Separate state for display

    # Move detection variables
    move_history = []
    captured_pieces = []  # Track captured pieces
    waiting_for_move = False
    move_start_state = None
    last_move_text = ""
    last_move_time = 0
    move_display_duration = 5.0  # Longer display for captures

    # Player turn tracking
    current_turn = "white (silver)"
    turn_text = "Silver's turn"

    print("📸 Camera running...")
    print("🟢 Show chessboard to calibrate...")
    print("Press 'b' to lock the board position")
    print("After locking, place pieces in initial position")
    print("Press 's' to START the game when pieces are set")
    print("Then each player presses 'm' after completing their move")
    print("Press 'r' to reset everything")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray : ", gray)
        cv2.imshow("edged", cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150))
        # Detect chessboard
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FILTER_QUADS +
            cv2.CALIB_CB_FAST_CHECK
        )
        key = cv2.waitKey(1) & 0xFF

        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            new_inner = corners.reshape(7, 7, 2)

            # If still inside calibration window
            if not board_locked:
                inner_corners = new_inner
                full_corners = extrapolate_corners(inner_corners)
                squares, squares_info = extract_8x8_squares(frame, full_corners)
                board_state = analyze_board_state(squares, squares_info, 4)
                for square_name, square_img in squares.items():
                    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    # Apply edge detection
                    edges = cv2.Canny(blur, 50, 150)
                    # cv2.imshow(f"edges de {square_name}: ",edges)
                    # Calculate edge density (pieces have more edges)
                    contours, hierarchies = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    if maximum_contours< len(contours):
                        maximum_contours = len(contours)

                current_display_state = board_state.copy()  # Update display state
                display_board_state(board_state)
                display_frame = draw_chessboard_grid(display_frame, full_corners, current_display_state)

                if key == ord('b'):
                    board_locked = True
                    print("🔒 Board position locked!")
                    print("Now place pieces in their initial positions...")
                    print("Press 's' when ready to start the game")

        # ---- After Lock ----
        if board_locked and full_corners is not None:
            squares, squares_info = extract_8x8_squares(frame, full_corners)

            current_time = time.time()
            if current_time - last_analysis_time > analysis_interval:
                # Always get the latest board state from camera
                latest_board_state = analyze_board_state(squares, squares_info,maximum_contours)

                if not game_started:
                    # In setup mode - update display to show current piece positions
                    board_state = latest_board_state.copy()
                    current_display_state = board_state.copy()
                elif waiting_for_move:
                    # In game mode, waiting for player to press 'm'
                    # Update display to show current positions, but don't change game state yet
                    current_display_state = latest_board_state.copy()
                else:
                    # Game started but not waiting for move (shouldn't happen)
                    current_display_state = latest_board_state.copy()

                last_analysis_time = current_time

            # Always draw the grid with current display state
            display_frame = draw_chessboard_grid(display_frame, full_corners, current_display_state)

            # Display different status based on game state
            y_offset = 160

            if not game_started:
                # Setup mode
                status_text = "SETUP MODE - Place pieces in initial position"
                cv2.putText(display_frame, status_text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                y_offset += 30
                cv2.putText(display_frame, "Press 's' when ready to start game",
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Game mode
                # Draw turn indicator
                turn_color = (255, 255, 255) if current_turn == "white (silver)" else (255, 215, 0)  # Gold for blue
                cv2.putText(display_frame, turn_text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, turn_color, 2)
                y_offset += 30

                # Draw waiting indicator
                if waiting_for_move:
                    cv2.putText(display_frame, f"Make your move, then press 'm'",
                                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(display_frame, f"Press 'm' when you complete your move",
                                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Draw last move with capture info
                if last_move_text and (current_time - last_move_time) < move_display_duration:
                    cv2.putText(display_frame, last_move_text, (20, y_offset + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Show captured pieces if any
                    if captured_pieces:
                        silver_captured = captured_pieces.count('white (silver)')
                        gold_captured = captured_pieces.count('blue (gold)')
                        captured_text = f"Captured: Silver {silver_captured} | Gold {gold_captured}"
                        cv2.putText(display_frame, captured_text, (20, y_offset + 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Piece statistics from current display state
            silver_count = sum(1 for info in current_display_state.values()
                               if info['has_piece'] and info['color'] == 'white (silver)')
            gold_count = sum(1 for info in current_display_state.values()
                             if info['has_piece'] and info['color'] == 'blue (gold)')

            stats_text = f"Silver: {silver_count} | Gold: {gold_count}"
            cv2.putText(display_frame, stats_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Move count
            cv2.putText(display_frame, f"Moves: {len(move_history)}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            status = "BOARD LOCKED" + (" - GAME ACTIVE" if game_started else " - SETUP MODE")
            color = (0, 255, 0) if game_started else (255, 165, 0)
        else:
            status = "CALIBRATING - Show chessboard"
            color = (0, 165, 255)

        # Display status
        cv2.putText(display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display instructions
        if not board_locked:
            instructions = ["'b': Lock board position"]
        elif not game_started:
            instructions = ["'s': Start game", "'r': Reset"]
        else:
            player_name = "Silver" if current_turn == "white (silver)" else "Gold"
            instructions = [
                f"'m': {player_name} completed move",
                "'r': Reset everything",
                "'q': Quit"
            ]

        for i, instr in enumerate(instructions):
            cv2.putText(display_frame, instr, (display_frame.shape[1] - 300, 40 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Chessboard Piece Detection", display_frame)

        # Handle key presses
        if key == ord('q'):
            break

        elif key == ord('r'):
            # Full reset
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
            print("🔄 Full reset. Show board again for calibration.")

        elif key == ord('s') and board_locked and not game_started:
            # Start the game with current piece positions as initial state
            game_started = True
            waiting_for_move = True
            # Store the current board state as the starting position
            board_state = current_display_state.copy()
            move_start_state = board_state.copy()
            captured_pieces = []  # Reset captured pieces
            print("🎮 Game started! Silver's turn. Make your move and press 'm' when done.")
            print(
                f"Initial position recorded with {sum(1 for info in board_state.values() if info['has_piece'])} pieces")

        elif key == ord('m') and board_locked and game_started:
            # Player signals move completion
            if waiting_for_move:
                print("\n--- Checking for move ---")
                print(
                    f"Move start state has {sum(1 for info in move_start_state.values() if info['has_piece'])} pieces")
                print(
                    f"Current state has {sum(1 for info in current_display_state.values() if info['has_piece'])} pieces")

                # Detect the move with capture information
                from_square, to_square, piece_color, captured_color, is_valid = detect_move(move_start_state,
                                                                                            current_display_state)

                if is_valid and piece_color == current_turn:
                    # Valid move detected
                    if captured_color:
                        # This was a capture
                        captured_name = "Gold" if captured_color == "blue (gold)" else "Silver"
                        move_text = f"{piece_color} {from_square} -> {to_square} captures {captured_name}"
                        captured_pieces.append(captured_color)
                        print(f"✅ Capture! {piece_color} captured {captured_name} piece")
                    else:
                        move_text = f"{piece_color} moved: {from_square} -> {to_square}"
                        print(f"✅ Move detected: {from_square} -> {to_square}")
                        #chess_engine()

                    last_move_text = move_text
                    last_move_time = time.time()

                    # Add to history with capture info
                    move_history.append({
                        'from': from_square,
                        'to': to_square,
                        'piece': piece_color,
                        'captured': captured_color,
                        'timestamp': time.time()
                    })

                    print(f"📝 Move recorded: {move_text}")

                    # Update the official board state to current position
                    board_state = current_display_state.copy()

                    # Switch turns
                    if current_turn == "white (silver)":
                        current_turn = "blue (gold)"
                        turn_text = "Gold's turn"
                    else:
                        current_turn = "white (silver)"
                        turn_text = "Silver's turn"

                    # Update start state for next move
                    move_start_state = board_state.copy()

                    # Keep waiting for next move
                    waiting_for_move = True

                elif is_valid:
                    print(f"❌ Wrong turn! It's {turn_text}")
                    print(f"Detected piece color: {piece_color}, Current turn: {current_turn}")
                else:
                    print("❌ No valid move detected. Did you complete your move?")
                    print("Make your move and try pressing 'm' again")

                    # Debug: Show piece counts
                    silver_now = sum(1 for info in current_display_state.values()
                                     if info['has_piece'] and info['color'] == 'white (silver)')
                    gold_now = sum(1 for info in current_display_state.values()
                                   if info['has_piece'] and info['color'] == 'blue (gold)')
                    print(f"Current board - Silver: {silver_now}, Gold: {gold_now}")
            else:
                print("⚠️ Already waiting for move completion")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Program terminated")
    print(f"📊 Total moves recorded: {len(move_history)}")
    print(
        f"📊 Captured pieces: Silver {captured_pieces.count('white (silver)')} | Gold {captured_pieces.count('blue (gold)')}")


if __name__ == "__main__":
    main()