# server.py
import pyarrow.flight as flight
import pyarrow as pa

class MyFlightServer(flight.FlightServerBase):
    def do_put(self, context, descriptor, reader, writer):
        # Example: Read the stream and collect data into a Pandas DataFrame
        table = reader.read_all()
        pandas_df = table.to_pandas()
        print(pandas_df.shape[0])

def start_server():
    server = MyFlightServer(('0.0.0.0', 47470))  # Listen on all interfaces
    print("Starting server on port 8815")
    server.serve()

if __name__ == "__main__":
    start_server()
