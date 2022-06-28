from sqlite3 import connect
from json import dumps, loads


def _create_table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS results (config TEXT PRIMARY KEY, data TEXT)")


def save_results(results, config):
    with connect("result_storage.sqlite3") as conn:
        _create_table(conn)
        conn.execute("INSERT INTO results (config, data) VALUES (?, ?)", (dumps(config), dumps(results)))


def get_cache(config):
    with connect("result_storage.sqlite3") as conn:
        _create_table(conn)
        cursor = conn.execute("SELECT data FROM results WHERE config=?", (dumps(config),))
        result = cursor.fetchone()
        if result is None:
            return None
        return loads(result[0])


def blow_cache():
    with connect("result_storage.sqlite3") as conn:
        conn.execute("DELETE FROM results")
