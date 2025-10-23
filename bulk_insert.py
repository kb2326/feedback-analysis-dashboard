import re
import csv
import io
from datetime import datetime
from urllib.parse import quote_plus
import sqlalchemy
from sqlalchemy.sql import text

# --- DB CONFIG ---
def get_db_config():
    """
    Database configuration - use environment variables in production.
    Update these values with your actual database credentials.
    """
    import os
    
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'your_username'),
        'password': os.getenv('DB_PASSWORD', 'your_password'),
        'database': os.getenv('DB_NAME', 'feedback_db'),
        'port': int(os.getenv('DB_PORT', 3306))
    }

def create_db_connection():
    try:
        config = get_db_config()
        encoded_password = quote_plus(config['password'])
        conn_str = f"mysql+pymysql://{config['user']}:{encoded_password}@{config['host']}:{config['port']}/{config['database']}"
        return sqlalchemy.create_engine(conn_str)
    except Exception as e:
        print(f"[ERROR] DB Connection: {e}")
        return None

# --- PARSE SQL FILE ---
def parse_insert_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'VALUES' in content:
        content = content.split('VALUES', 1)[-1]

    matches = re.findall(r'\((.*?)\)', content, re.DOTALL)
    cleaned_rows = []

    for raw_row in matches:
        cleaned_raw = raw_row.replace("`", "").replace("NULL", "NULL_PLACEHOLDER")
        fake_csv = io.StringIO(cleaned_raw)
        reader = csv.reader([fake_csv.getvalue()], skipinitialspace=True, quotechar="'", escapechar='\\')
        parsed = next(reader)

        cleaned = []
        for val in parsed:
            val = val.strip()
            if val in ["NULL_PLACEHOLDER", "", "X''", "0000-00-00 00:00:00","X'6E756C6C'","null"]:
                cleaned.append(None)
            elif val.startswith("X'") and val.endswith("'"):
                try:
                    cleaned.append(bytes.fromhex(val[2:-1]).decode('utf-8'))
                except:
                    cleaned.append(None)
            elif val.startswith("'") and val.endswith("'"):
                cleaned.append(val[1:-1])
            elif re.match(r'^-?\d+(\.\d+)?$', val):
                cleaned.append(float(val) if '.' in val else int(val))
            else:
                cleaned.append(val)

        while len(cleaned) < 19:
            cleaned.append(None)

        if len(cleaned) == 19:
            cleaned_rows.append(cleaned)
        else:
            print(f"[WARN] Skipped row (col count = {len(cleaned)}): {cleaned[:3]}")
    return cleaned_rows

# --- BULK INSERT ---
def bulk_insert_feedback(engine, rows):
    columns = [
        'id', 'user_id', 'course_id', 'client_id', 'client_name', 'client_type',
        'client_sector', 'country', 'course_name', 'course_status', 'course_duration',
        'course_category', 'rating', 'got_what_you_needed', 'what_did_you_like',
        'what_could_be_improved', 'how_many_hours_it_took', 'last_updated',
        'datetime_originally_submitted'
    ]

    success, failed = 0, 0
    failed_rows = []

    with engine.begin() as conn:
        query = text(f"""
            INSERT INTO feedback ({', '.join(columns)})
            VALUES ({', '.join([f':{col}' for col in columns])})
            ON DUPLICATE KEY UPDATE
            {', '.join([f"{col}=VALUES({col})" for col in columns if col != 'id'])}
        """)

        for i, row in enumerate(rows):
            try:
                for idx in [0, 1, 2, 3]:
                    row[idx] = int(row[idx]) if row[idx] is not None else None
                for idx in [10, 12, 16]:
                    if row[idx] is not None:
                        row[idx] = float(row[idx]) if '.' in str(row[idx]) else int(row[idx])
                for dt in [17, 18]:
                    if isinstance(row[dt], str) and row[dt] == "0000-00-00 00:00:00":
                        row[dt] = None

                conn.execute(query, dict(zip(columns, row)))
                success += 1
            except Exception as e:
                failed += 1
                failed_rows.append((i + 1, e, row.copy()))
    return success, failed, failed_rows

# --- MAIN RUN ---
def run_bulk_insert_from_sql_file(sql_file_path):
    engine = create_db_connection()
    if not engine:
        print("[ERROR] Could not connect to DB.")
        return

    print("[INFO] Reading SQL file...")
    rows = parse_insert_file(sql_file_path)
    print(f"[INFO] Parsed rows: {len(rows)}")

    print("[INFO] Starting import...")
    success, failed, failed_details = bulk_insert_feedback(engine, rows)

    print(f"\n[RESULT] ✅ {success} rows inserted.")
    print(f"[RESULT] ❌ {failed} rows failed.")

    if failed_details:
        with open("insert_failures.log", "w", encoding="utf-8") as log:
            for row_num, error, row_data in failed_details:
                log.write(f"Row {row_num} Error: {error}\n")
                log.write(f"Row Data: {row_data}\n\n")
        print("[INFO] Failed rows written to 'insert_failures.log'.")


# --- ENTRY POINT ---
if __name__ == "__main__":
    run_bulk_insert_from_sql_file("course-feedback.sql")
