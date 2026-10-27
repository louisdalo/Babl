"""A set of function to create custom progress bars"""

##### IMPORTS ##### ------------------------------------------------------------
import sys
import time
import psutil
import threading
from contextlib import contextmanager


##### PARAMETERS ##### ---------------------------------------------------------
BAR_WIDTH = 20              # Number of characters for the progress bar
REFRESH = 1                 # seconds between redraws (interactive terminal)
LOG_EVERY = 15              # seconds between lines when output is redirected


##### HELPERS ##### ------------------------------------------------------------
def _hms(seconds):
    """
        Formats a number of second into a string HH:MM:SS or MM:SS.

        Parameters
        ----------
        seconds: a numeric
            A number of seconds (decimals are removed).

        Returns
        ------
        A string
    """
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _gb(n_bytes):
    """
        Formats a number of bytes into GB or MB.

        Parameters
        ----------
        n_bytes: a numeric
            A number of bytes.

        Returns
        ------
        A string
    """
    return f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.0f} MB"

def _bytes_read(proc):
    """
        Outputs the total number of bytes this process has read so far 
        (None if unsupported).

        Parameters
        ----------
        proc: psutil.Process() object or None.

        Returns
        ------
        A number of bytes
    """
    try:
        io = proc.io_counters()
        return getattr(io, "read_chars", io.read_bytes)
    except (AttributeError, NotImplementedError, psutil.Error):
        return None


##### PROGRESS ##### -----------------------------------------------------------
@contextmanager 
def progress(label: str, total_bytes=None):
    """
        Shows a live progress while a long file loading runs in the main thread.

        How it works: a background thread measures how many bytes the current 
        process has read since its start and compares it to `total_bytes` 
        (the compressed size of the file being scanned). This gives a 
        percentage, a speed and an ETA. 
        
        Without `total_bytes` (or without psutil), it only shows a spinner
        with the elapsed time.

        Parameters
        ----------
        label: a string
            The label to display on the progress bar description.
        total_bytes: a numeric
            The compressed size (in bytes) of the file being scanned.       

        Usage
        -----
            with progress("[SCAN 1/3] Filtering", dump.stat().st_size):
                con.execute(...)
    """
    proc = psutil.Process() if psutil else None
    start_bytes = _bytes_read(proc) if proc else None
    use_bytes = bool(total_bytes) and start_bytes is not None
    interactive = sys.stdout.isatty()
    start = time.time()
    stop = threading.Event()
    state = {"last_len": 0}

    def render(final=False):
        elapsed = time.time() - start
        if use_bytes:
            done = min(max(_bytes_read(proc) - start_bytes, 0), total_bytes)
            frac = 1.0 if final else min(done / total_bytes, 0.999)
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total_bytes - done) / speed if speed > 0 else 0
            filled = int(BAR_WIDTH * frac)
            bar = "#" * filled + "-" * (BAR_WIDTH - filled)
            line = (f"{label} [{bar}] {frac:6.1%} | "
                    f"{_gb(done if not final else total_bytes)}/{_gb(total_bytes)} read | "
                    f"{speed / 1e6:.0f} MB/s | elapsed {_hms(elapsed)}")
            if not final:
                line += f" | ETA {_hms(eta)}"
        else:
            spin = "|/-\\"[int(elapsed * 2) % 4]
            line = f"{label} {'done' if final else spin} | elapsed {_hms(elapsed)}"
        return line

    def loop():
        next_log = LOG_EVERY
        while not stop.wait(REFRESH):
            if interactive:
                line = render()
                pad = " " * max(state["last_len"] - len(line), 0)
                sys.stdout.write("\r" + line + pad)
                sys.stdout.flush()
                state["last_len"] = len(line)
            elif time.time() - start >= next_log:
                print(render(), flush=True)
                next_log += LOG_EVERY

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        line = render(final=True)
        if interactive:
            pad = " " * max(state["last_len"] - len(line), 0)
            sys.stdout.write("\r" + line + pad + "\n")
        else:
            line_out = line
            print(line_out)
        sys.stdout.flush()

def track(iterable, label, total=None, unit="items"):
    """
        Shows a progress bar for a Python loop whose length is known 
        (UNLIKE `progress`, which follows bytes read by a blocking call).

        Parameters
        ----------
        iterable: an iterable object
            The object being looped on.
        label: a string
            The label to display on the progress bar description.
        total: a numeric
            The number of iterations in the loop.      

        Usage
        -----
            for row in track(rows, "  descriptions", unit="works"):
                ...

        Shows a bar, count, speed and ETA. Like `progress`, it redraws in place
        on a terminal and prints a line every LOG_EVERY seconds when the output
        is redirected. `total` defaults to len(iterable).

    """
    if total is None:
        total = len(iterable)
    interactive = sys.stdout.isatty()
    start = last_draw = time.time()
    next_log = LOG_EVERY
    last_len = 0
    n = 0

    def render(final=False):
        elapsed = time.time() - start
        frac = 1.0 if final or not total else min(n / total, 1.0)
        filled = int(BAR_WIDTH * frac)
        bar = "#" * filled + "-" * (BAR_WIDTH - filled)
        speed = n / elapsed if elapsed > 0 else 0
        line = (f"{label} [{bar}] {frac:6.1%} | {n:,}/{total:,} {unit} | "
                f"{speed:,.0f}/s | elapsed {_hms(elapsed)}")
        if not final and speed > 0:
            line += f" | ETA {_hms((total - n) / speed)}"
        return line

    try:
        for item in iterable:
            yield item
            n += 1
            now = time.time()
            if interactive and now - last_draw >= 0.25:
                line = render()
                sys.stdout.write("\r" + line + " " * max(last_len - len(line), 0))
                sys.stdout.flush()
                last_len, last_draw = len(line), now
            elif not interactive and now - start >= next_log:
                print(render(), flush=True)
                next_log += LOG_EVERY
    finally:
        line = render(final=n >= total)
        if interactive:
            sys.stdout.write("\r" + line + " " * max(last_len - len(line), 0) + "\n")
        else:
            print(line)
        sys.stdout.flush()

@contextmanager
def fraction_progress(label: str):
    """
        Shows a progress bar for a task that reports its own progress as a
        fraction (e.g. utils.embeddings.embed_books). Yields a callback
        `update(fraction, message)`. Like `progress`, it redraws in place on a
        terminal and prints a line every LOG_EVERY seconds when the output is
        redirected.

        Parameters
        ----------
        label: a string
            The label to display on the progress bar description.

        Usage
        -----
            with fraction_progress("  embeddings") as update:
                embed_books(books, progress=update)
    """
    interactive = sys.stdout.isatty()
    start = last_draw = time.time()
    state = {"frac": 0.0, "msg": "", "last_len": 0, "next_log": LOG_EVERY}

    def render(final=False):
        elapsed = time.time() - start
        frac = 1.0 if final else min(max(state["frac"], 0.0), 1.0)
        filled = int(BAR_WIDTH * frac)
        bar = "#" * filled + "-" * (BAR_WIDTH - filled)
        msg = "done" if final else state["msg"]
        line = f"{label} [{bar}] {frac:6.1%} | {msg} | elapsed {_hms(elapsed)}"
        if not final and frac > 0.02:
            line += f" | ETA {_hms(elapsed * (1 - frac) / frac)}"
        return line

    def update(fraction, message):
        nonlocal last_draw
        state["frac"], state["msg"] = fraction, message
        now = time.time()
        if interactive and now - last_draw >= 0.25:
            line = render()
            sys.stdout.write("\r" + line + " " * max(state["last_len"] - len(line), 0))
            sys.stdout.flush()
            state["last_len"], last_draw = len(line), now
        elif not interactive and now - start >= state["next_log"]:
            print(render(), flush=True)
            state["next_log"] += LOG_EVERY

    try:
        yield update
    finally:
        line = render(final=True)
        if interactive:
            sys.stdout.write("\r" + line + " " * max(state["last_len"] - len(line), 0) + "\n")
        else:
            print(line)
        sys.stdout.flush()