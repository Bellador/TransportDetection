import os
import sys
import psycopg2
from psycopg2 import sql


class DbQuerier:
    path_db_psw = "./db_psw.txt"

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
                # conn = psycopg2.connect("host=127.0.0.1 dbname=osm_geneva user=postgres port=5432 password={}".format(password))
                conn = psycopg2.connect("host=127.0.0.1 dbname=osm_paris_streets_buildings user=postgres port=5432 password={}".format(password))
                return conn
            except Exception as e:
                print(f"DB connection error {e}. Try again.")
                sys.exit(1)

    def spatial_db_query(self, geoparsing_str, mode='OCR'):
        '''
        function that performs the spatial queries to the database for OCR and TEXT extracted (potential) locations

        :param geoparsing_str:
        :param mode:
        :return:
        '''
        # escape single quotes by doubling them (since they can have meaning in a string they are not removed!)
        single_quote_indexes = []
        escaped_geoparsing_str = list(geoparsing_str)
        for index, letter in enumerate(geoparsing_str):
            if letter == "'":
                single_quote_indexes.append(index)
        # index increases if multiple ' are found!
        for count, i in enumerate(single_quote_indexes):
            escaped_geoparsing_str.insert(i + count, "'")
        # convert back to string
        escaped_geoparsing_str = ''.join(escaped_geoparsing_str)
        # levenshtein only allows max. 255 chars
        if len(escaped_geoparsing_str) < 256:
            with self.conn.cursor() as cursor:
                try:
                    # differentiates between modes to easily modify queries depending on origin
                    if mode == 'OCR':
                        query = sql.SQL(f'''select q1.name, q2.ST_AsText
                            from (select name from paris_osm_streets_buildings where levenshtein_less_equal(LOWER(name), LOWER('{escaped_geoparsing_str}'), 2) <= 2) q1 cross join
                            (select ST_AsText(ST_UNION(ARRAY(select wkb_geometry from paris_osm_streets_buildings where levenshtein_less_equal(LOWER(name), LOWER('{escaped_geoparsing_str}'), 2) <= 2 group by wkb_geometry)))) q2 limit 1;'''). \
                            format(escaped_geoparsing_str=sql.Identifier(escaped_geoparsing_str))

                    elif mode == 'TEXT':
                        query = sql.SQL(f'''select q1.name, q2.ST_AsText
                            from (select name from paris_osm_streets_buildings where levenshtein_less_equal(LOWER(name), LOWER('{escaped_geoparsing_str}'), 1) <= 1) q1 cross join
                            (select ST_AsText(ST_UNION(ARRAY(select wkb_geometry from paris_osm_streets_buildings where levenshtein_less_equal(LOWER(name), LOWER('{escaped_geoparsing_str}'), 1) <= 1 group by wkb_geometry)))) q2 limit 1;'''). \
                            format(escaped_geoparsing_str=sql.Identifier(escaped_geoparsing_str))

                    else:
                        print(f'[!] Invalid db_query mode={mode}. Exit program')
                        exit()
                    cursor.execute(query)
                    result = cursor.fetchall()
                    return result
                except Exception as e:
                    print(f'\n[!] Transaction error: {e} \n [!] rollback.')
                    self.conn.rollback()
                    return None
        else:
            print(f'[!] Geoparse str exceeds 255 chars')