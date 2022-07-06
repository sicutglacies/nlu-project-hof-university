from neo4j import GraphDatabase

import logging
import os
from neo4j.exceptions import ServiceUnavailable


import numpy as np

class App:


    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))


    def close(self):
        self.driver.close()

    def set_constraint(self):
        with self.driver.session() as session:
            result = session.write_transaction(self._set_constraint)

    @staticmethod
    def _set_constraint(tx):
        query = ("CREATE CONSTRAINT unique_index IF NOT EXISTS FOR (c:Citation) REQUIRE c.id IS UNIQUE")
        try:
            result = tx.run(query)
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(

            query=query, exception=exception))

            raise

    def load_data(self):
        data = np.load('../data/citation_relations.npy', allow_pickle=True)
        # data = [{'1': ['2']}, {'2': ['3']}, {'3': ['2']}]
        count = 1
        with self.driver.session() as session:
            for i in data:
                result = session.write_transaction(self._create_citation, i)
                count += 1
                if (count > 500):
                    break

    @staticmethod
    def _create_citation(tx, i):
        keys = list(i.keys())
        if (len(keys)):
            id = keys[0]

            query = ("CREATE (c:Citation { id: $id })")

            try:
                query1 = ("MATCH (c:Citation) WHERE c.id = $id RETURN c")

                result = tx.run(query1, id=id)
                data = [record for record in result]

                if (len(data) == 0):
                    result = tx.run(query, id=id)

                if (len(i[id])):
                    for i in i[id]:

                        query1 = ("MATCH (c:Citation) WHERE c.id = $id RETURN c")

                        result = tx.run(query1, id=i)
                        data = [record for record in result]
                        if (len(data) == 0):
                            query = ('MATCH (c:Citation) WHERE c.id = "' + id + '" CREATE (c1:Citation { id: $id }) CREATE (c1)-[:CITING]->(c)')
                            result = tx.run(query, id=i)
                        else:
                            query = ('MATCH (c:Citation) WHERE c.id = "' + id + '" MATCH (c1:Citation) WHERE c1.id = $id CREATE (c1)-[:CITING]->(c)')
                            result = tx.run(query, id=i)

            except ServiceUnavailable as exception:
                logging.error("{query} raised an error: \n {exception}".format(

                query=query, exception=exception))

                raise
    	

if __name__ == "__main__":

    uri = "neo4j+s://192b0607.databases.neo4j.io"

    user = "neo4j"

    password = os.env['DB_PASS']

    app = App(uri, user, password)
    app.set_constraint()
    app.load_data()

    app.close()

# By Nurbak A
