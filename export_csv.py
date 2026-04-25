# export from database to csv file
import sqlite3
import csv
import os
from pathlib import Path


def export_table_to_csv(conn, table_name, output_file):
    cursor = conn.cursor()
    # only export the highest 1000 frequency rows to save time and space
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY frequency DESC LIMIT 10000")
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]

    # change id columns to token for bigrams and trigrams
    if table_name == "bigrams":
        for i in range(len(rows)):
            _, token1_id, token2_id, frequency = rows[i]
            cursor.execute("SELECT token FROM unigrams WHERE id = ?", (token1_id,))
            token1 = cursor.fetchone()[0]
            cursor.execute("SELECT token FROM unigrams WHERE id = ?", (token2_id,))
            token2 = cursor.fetchone()[0]
            rows[i] = (token1, token2, frequency)
        column_names[0] = "token1"
        column_names[1] = "token2"
    elif table_name == "trigrams":
        for i in range(len(rows)):
            _, token1_id, token2_id, token3_id, frequency = rows[i]
            cursor.execute("SELECT token FROM unigrams WHERE id = ?", (token1_id,))
            token1 = cursor.fetchone()[0]
            cursor.execute("SELECT token FROM unigrams WHERE id = ?", (token2_id,))
            token2 = cursor.fetchone()[0]
            cursor.execute("SELECT token FROM unigrams WHERE id = ?", (token3_id,))
            token3 = cursor.fetchone()[0]
            rows[i] = (token1, token2, token3, frequency)
        column_names[0] = "token1"
        column_names[1] = "token2"
        column_names[2] = "token3"

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)  # Write header
        writer.writerows(rows)         # Write data rows


def print_table(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY frequency DESC LIMIT 20")
    rows = cursor.fetchall()
    for row in rows:
        print(row)


def main():
    db_path = "./ngrams/ngrams.db"
    output_dir = Path("./exported_csv")
    output_dir.mkdir(exist_ok=True)

    with sqlite3.connect(str(db_path)) as conn:
        export_table_to_csv(conn, "unigrams", output_dir / "unigrams.csv")
        export_table_to_csv(conn, "bigrams", output_dir / "bigrams.csv")
        export_table_to_csv(conn, "trigrams", output_dir / "trigrams.csv")

        print_table(conn, "unigrams")
        print_table(conn, "bigrams")
        print_table(conn, "trigrams")


if __name__ == "__main__":
    main()