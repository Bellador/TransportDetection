import os
import sys
import psycopg2
from psycopg2 import sql


class DbQuerier:
    # path_db_psw = r"C:\Users\mhartman\PycharmProjects\YoutubeAPI-Query\db_psw.txt"
    path_db_psw = "../YoutubeAPI-Query/db_psw.txt"

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
                # conn = psycopg2.connect("host=127.0.0.1 dbname=osm_paris_streets_buildings user=postgres port=5432 password={}".format(password))
                conn = psycopg2.connect("host=127.0.0.1 dbname=osm user=postgres port=5432 password={}".format(password))
                return conn
            except Exception as e:
                print(f"Error {e}. Try again.")
                sys.exit(1)

    def db_query(self, geoparsing_str):
        # escape single quotes by doubling them (since they can have meaning in a string they are not removed!)
        single_quote_indexes = []
        escaped_geoparsing_str = list(geoparsing_str)
        for index, letter in enumerate(geoparsing_str):
            if letter == "'":
                single_quote_indexes.append(index)
        # index increases if mulitple ' are found!
        for count, i in enumerate(single_quote_indexes):
            escaped_geoparsing_str.insert(i + count, "'")
        # convert back to string
        escaped_geoparsing_str = ''.join(escaped_geoparsing_str)

        with self.conn.cursor() as cursor:
            try:
                query = sql.SQL(f'''SELECT name, ST_AsText(wkb_geometry) FROM paris_osm_streets_buildings as x WHERE x.name = '{escaped_geoparsing_str}';''').\
                    format(escaped_geoparsing_str=sql.Identifier(escaped_geoparsing_str))
                # check
                # print(f'\r{query.as_string(self.conn)}', end='')
                # execute query
                cursor.execute(query)
                result = cursor.fetchall()
                return result
            except Exception as e:
                print(f'\n[!] Transaction error: {e} \n [!] rollback.')
                print(f'\n[!] String: {geoparsing_str}')
                self.conn.rollback()
                return None
