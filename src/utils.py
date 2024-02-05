import threading

class ThreadUtils:
    def __init__(self):
        self.print_lock = threading.Lock()
        self.save_lock = threading.Lock()

    def thread_safe_print(self, message):
        with self.print_lock:
            print(message)

    def acquire_save_lock(self):
        self.save_lock.acquire()

    def release_save_lock(self):
        self.save_lock.release()