import time
from queue import Empty

class ClassifierWorker:
    """
    Background worker for processing queued crops in batches.
    Designed to run in its own thread.
    """

    def __init__(
        self,
        queue,
        cnn,
        card_db,
        seen_cards,
        overlay_lock,
        live_detections,
        batch_size,
        stop_event,
        queue_wait_times=None,
    ):
        self.queue = queue
        self.cnn = cnn
        self.card_db = card_db
        self.seen_cards = seen_cards
        self.overlay_lock = overlay_lock
        self.live_detections = live_detections
        self.batch_size = batch_size
        self.stop_event = stop_event
        self.clf_time = 0.0
        self.queue_wait_times = queue_wait_times if queue_wait_times is not None else []

    def run(self, budget=float("inf")):
        """Process queued crops in batches within `budget` seconds."""
        start = time.time()
        while (time.time() - start) < budget and not self.queue.empty():
            imgs = []
            cats = []
            queue_waits = []
            for _ in range(min(self.batch_size, self.queue.qsize())):
                try:
                    crop, cat, enqueue_time = self.queue.get_nowait()
                    queue_wait = time.time() - enqueue_time
                    queue_waits.append(queue_wait)
                except Empty:
                    break
                imgs.append(crop)
                cats.append(cat)
            if not imgs:
                break
            t0 = time.time()
            labels = self.cnn.classify(imgs, cats)
            self.clf_time += time.time() - t0
            if self.queue_wait_times is not None and queue_waits:
                self.queue_wait_times.extend(queue_waits)
            for card_id in labels:
                if card_id not in self.seen_cards:
                    self.seen_cards.add(card_id)
                    name = self.card_db.get_name_by_seriesid_id(*card_id.split(" "))
                    display_str = f"{card_id} {name}" if name else card_id
                    with self.overlay_lock:
                        self.live_detections[card_id] = (display_str, time.time())

    def loop(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            if self.queue.empty():
                time.sleep(0.01)
                continue
            self.run()
