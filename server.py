# server.py
import pyarrow.flight as flight
import pyarrow as pa
import pandas as pd
from embed import Embedder
import time

class MyFlightServer(flight.FlightServerBase):

    def __init__(self, location, **kwargs):
        super().__init__(location, **kwargs)
        self.embeddings= None  # Instance variable to store embeddings
        self.embeddingcompletion = False


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
        self.embeddings = embedder.embed(text_data)
        print("Embedding generation done")
        self.embeddingcompletion = True
    
    def process_data_demo(self, pandas_df):
        #text_data = pandas_df['message'].tolist()
        def add_exclamation(message):
            return message + '!!!'
        # Apply the function
        pandas_df['message'] = pandas_df['message'].apply(add_exclamation)
        self.processed_df = pandas_df
    def do_put(self, context, descriptor, reader, writer):
        # Example: Read the stream and collect data into a Pandas DataFrame
        table = reader.read_all()
        # self.getEmbeddings(pandas_df)
        self.getEmbeddings(table)
    
    def do_get(self, context, ticket):
        # Example data you want to send back
        # embeddings_df = pd.DataFrame(self.embeddings)
        embeddings_table = pa.Table.from_arrays([pa.array(self.embeddings)], names=['embeddings'])
        # table = pa.Table.from_pandas(embeddings_table)
        # Send the table back to the client
        return flight.RecordBatchStream(embeddings_table)

def start_server():
    server = MyFlightServer(('0.0.0.0', 5111))  # Listen on all interfaces
    print("Starting server on port 5111")
    server.serve()

if __name__ == "__main__":
    start_server()
