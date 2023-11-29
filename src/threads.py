import threading

print_lock = threading.Lock()

def thread_safe_print(message):
    with print_lock:
        print(message)