"""
chess_server.py
Run this to start everything:  python chess_server.py
Opens http://localhost:5000 in your browser.
"""

from flask import Flask, render_template
from flask_socketio import SocketIO
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chess_bridge_secret'

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=False,
    engineio_logger=False
)


@app.route('/')
def index():
    return render_template('chess_ui.html')


@socketio.on('connect')
def on_connect():
    print('🌐 Browser connected')

@socketio.on('disconnect')
def on_disconnect():
    print('🌐 Browser disconnected')

@socketio.on('client_ready')
def on_ready(data):
    print('✅ UI ready')


# ── API called from main.py ──

def _emit(event, data):
    try:
        socketio.emit(event, data, namespace='/')
        print(f'📡 emitted {event} → {data}')
    except Exception as e:
        print(f'📡 emit FAILED {event}: {e}')

def push_move(from_sq: str, to_sq: str, player: str = 'silver'):
    _emit('apply_move', {'from': from_sq, 'to': to_sq, 'player': player})

def push_status(msg: str):
    _emit('status', {'msg': msg})

def push_eval(score: float):
    _emit('eval', {'score': round(score, 2)})

def push_reset():
    _emit('reset_game', {})

def push_sf_suggestion(uci: str):
    if len(uci) >= 4:
        _emit('sf_suggestion', {'from': uci[:2], 'to': uci[2:4]})


def start_cv_and_serve(host='0.0.0.0', port=5000):
    """Start CV loop in background thread, then run Flask in main thread."""
    from Main import main as cv_main

    # CV runs in background
    cv_thread = threading.Thread(target=cv_main, daemon=True)
    cv_thread.start()

    print(f'🌐 Chess UI → http://localhost:{port}')

    # Flask runs in main thread — this blocks
    socketio.run(
        app, host=host, port=port,
        debug=False, use_reloader=False, allow_unsafe_werkzeug=True
    )


if __name__ == '__main__':
    start_cv_and_serve()