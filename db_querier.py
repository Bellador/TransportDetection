import os
import sys
import psycopg2
from psycopg2 import sql


class DbQuerier:
    path_db_psw = "db_psw.txt"

    def __init__(self):
        self.conn = self.connect_db()

    def connect_db(self):
        while True:
            #check if db password file exists, otherwise manual entry
            if os.path.isfile(DbQuerier.path_db_psw):
                with open(DbQuerier.path_db_psw, 'r') as f:
                    password = f.read()
            else:
                password = input("Input database password: ")
            try:
                conn = psycopg2.connect("host=127.0.0.1 dbname=osm user=postgres port=5433 password={}".format(password))
                return conn
            except Exception as e:
                print(f"Error {e}. Try again.")
                sys.exit(1)

    def levenshtein_dist_query(self, geoparsing_str):
        # escape single quotes by doubling them (since they can have meaning in a string they are not removed!)
        single_quote_indexes = []
        escaped_geoparsing_str = geoparsing_str
        for index, letter in enumerate(escaped_geoparsing_str):
            if letter == "'":
                single_quote_indexes.append(index)
        for i in single_quote_indexes:
            escaped_geoparsing_str = escaped_geoparsing_str[:i].rstrip() + "'" + escaped_geoparsing_str[i:].lstrip()

        with self.conn.cursor() as cursor:
            try:
                query = sql.SQL(f'''SELECT name, ST_AsText(wkb_geometry) FROM geneva_streets as x WHERE levenshtein_less_equal(x.name, '{escaped_geoparsing_str}', 2) <= 2;''').\
                    format(escaped_geoparsing_str=sql.Identifier(escaped_geoparsing_str))
                # check
                print(f'\n{query.as_string(self.conn)}')
                # execute query
                cursor.execute(query)
                result = cursor.fetchall()
                return result
            except Exception as e:
                print(f'\n[!] {e}')