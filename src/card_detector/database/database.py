import sqlite3
from io import BytesIO
from PIL import Image

class CardDB:
    def __init__(self, db_path="data/cards.db", *, check_same_thread: bool = False):
        """Open a connection to the cards database.

        ``check_same_thread`` defaults to ``False`` so that the same connection
        can be safely used across multiple threads for read-only queries.
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row

    def get_name_by_seriesid_id(self, series_id, id_):
        c = self.conn.cursor()
        c.execute('SELECT name FROM cards WHERE series_id=? AND id=?', (series_id, id_))
        row = c.fetchone()
        return row['name'] if row else None
    
    def get_image_blob_by_seriesid_id(self, series_id, id_):
        c = self.conn.cursor()
        c.execute('SELECT image_blob FROM cards WHERE series_id=? AND id=?', (series_id, id_))
        row = c.fetchone()
        if row and row["image_blob"]:
            return Image.open(BytesIO(row["image_blob"]))
        return None

    def close(self):
        self.conn.close()

# Test usage
if __name__ == "__main__":
    db = CardDB()
    print(db.get_name_by_seriesid_id("A3", "1"))
    db.close()
