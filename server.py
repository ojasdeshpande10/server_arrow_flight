# server.py
import pyarrow.flight as flight
import pyarrow as pa
import pandas as pd
from embed import Embedder
import time
import numpy as np

class MyFlightServer(flight.FlightServerBase):

    def __init__(self, location, **kwargs):
        super().__init__(location, **kwargs)
        self.embeddings= None  # Instance variable to store embeddings
        self.embeddingcompletion = False
        self.tokenization_time = 0
        self.embedding_time = 0
    def do_exchange(self, context, descriptor, reader, writer):

        table = reader.read_all()
        pandas_df = table.to_pandas()
        # text_data = pandas_df['message'].tolist()
        # embedder = Embedder("roberta-base")
        # embeddings = embedder.embed(text_data)
        # embeddings_df = pd.DataFrame(embeddings)
        # print("Shape of embeddings")
        # print(embeddings_df.shape)
        # embeddings_df.columns = [f'feature_{i}' for i in range(embeddings_df.shape[1])]
        # print("Shape of embeddings")
        # print(embeddings_df.shape)
        pandas_df['new_column'] = pandas_df['column1'] + '_' + pandas_df['column2']
        table = pa.Table.from_pandas(pandas_df)
        print(table)
        writer.begin(table.schema)
        writer.write_table(table)
        print("Writing starting")
        writer.close()
        print("writing closed")

    def getEmbeddings(self, table):
        text_data = table.column('message').to_pylist()
        # put model in init
        embedder = Embedder("roberta-base")
        self.embeddings, tokenization_time, embedding_time = embedder.embed(text_data)
        self.tokenization_time = tokenization_time
        self.embedding_time = embedding_time    
        self.embeddingcompletion = True
    def do_put(self, context, descriptor, reader, writer):
        # Example: Read the stream and collect data into a Pandas DataFrame
        self.table = reader.read_all()
        self.getEmbeddings(self.table)
    
    def do_get(self, context, ticket):
        updated_table = self.table.append_column('embedding', pa.array(self.embeddings))
        df = updated_table.to_pandas()
        result_df = df.groupby('year_datetime').agg({
            'embedding': lambda x: np.mean(np.vstack([np.array(embedding) for embedding in x]), axis=0).tolist()
        }).reset_index()
        result_df = result_df.rename(columns={'year_datetime': 'user_id_yr_week'})
        result_table = pa.Table.from_pandas(result_df)
        tokenizing_time = pa.array([self.tokenization_time] * result_table.num_rows, type=pa.float64())
        embedding_time = pa.array([self.embedding_time] * result_table.num_rows)
        updated_table = result_table.append_column('tokenization_time', pa.array(tokenizing_time))
        updated_table = updated_table.append_column('embedding_time', pa.array(embedding_time))
        return flight.RecordBatchStream(updated_table)

def start_server():
    server = MyFlightServer(('0.0.0.0', 5111))  # Listen on all interfaces
    print("Starting server on port 5111")
    server.serve()

if __name__ == "__main__":
    start_server()
