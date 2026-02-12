import time
import cv2

from advanced_test import analyze_board_state, display_board_state, extract_8x8_squares, extrapolate_corners, \
    draw_chessboard_grid


def main():
    # Camera setup
    cap = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    pattern_size = (7, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # State variables
    board_locked = False
    # seconds

    inner_corners = None
    full_corners = None

    last_analysis_time = 0
    analysis_interval = 2.0
    board_state = {}

    print("📸 Camera running...")
    print("🟢 Show chessboard for 10 seconds to calibrate...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display_frame = frame.copy()
        cv2.imshow("HSV : ", cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray : ", gray)
        cv2.imshow("edged", cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150))

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
                board_state = analyze_board_state(squares, squares_info)
                display_board_state(board_state)

                display_frame = draw_chessboard_grid(display_frame, full_corners, board_state)
                if key == ord('b'):
                    board_locked = True
                    print("🔒 Board locked permanently!")

        # ---- After Lock ----
        if board_locked and full_corners is not None:

            squares, squares_info = extract_8x8_squares(frame, full_corners)

            current_time = time.time()
            if current_time - last_analysis_time > analysis_interval:
                board_state = analyze_board_state(squares, squares_info)
                display_board_state(board_state)
                last_analysis_time = current_time

            display_frame = draw_chessboard_grid(display_frame, full_corners, board_state)

            # Piece statistics
            white_count = sum(1 for info in board_state.values()
                              if info['has_piece'] and info['color'] == 'white')
            black_count = sum(1 for info in board_state.values()
                              if info['has_piece'] and info['color'] == 'black')

            stats_text = f"White: {white_count} | Black: {black_count}"
            cv2.putText(display_frame, stats_text, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            status = "BOARD LOCKED"
            color = (0, 255, 0)

        else:
            status = "CALIBRATING (Move board if needed)"
            color = (0, 165, 255)

        # Display status
        cv2.putText(display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Chessboard Piece Detection", display_frame)



        if key == ord('q'):
            break

        elif key == ord('r'):
            board_locked = False

            inner_corners = None
            full_corners = None
            board_state = {}
            print("🔄 Reset. Show board again for calibration.")
        for square_name in ["e4", "d5", "a1", "h8"]:
            cv2.imshow(f"Square {square_name}",squares[square_name] )

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Program terminated")

if __name__ == "__main__":
    main()