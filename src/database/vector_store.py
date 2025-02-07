from src.config.settings import AILogger, get_settings, Settings, OpenAISettings

import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import os

import pandas as pd
from dataclasses import dataclass, field


from openai import OpenAI
from timescale_vector import client

load_dotenv(find_dotenv())

ai_logger = AILogger()

@dataclass
class VectorStore:
    """Manages vectore operations and database interactions
       Intialize VectorStore with settings, OpenAI client, and Timescale Vector client
    """
    settings:Settings = field(default_factory=lambda: get_settings())
    openai_client:OpenAI = field(default_factory=lambda : OpenAI(api_key=os.getenv('OPENAI_API_KEY')))

    def __post_init__(self):
        self.embedding_model= self.settings.openai.embedding_model
        self.vector_settings = self.settings.vector_store
        self.vs_client = client.Sync(
            service_url=self.settings.database.service_url,
            table_name=self.settings.vector_store.table_name,
            num_dimensions=self.settings.vector_store.embedding_dimensions,
            time_partition_interval=self.settings.vector_store.time_parition_interval
        )
    
    def get_embedding(self, text:str)->List[float]:
         """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
         text = text.replace('\n', ' ')
         start_time = time.time()
         embedding = self.openai_client.embeddings.create(input=[text], model=self.embedding_model).data[0].embedding
         elapsed_time = time.time()
         ai_logger.info(f'Generated embedding for "{text}" in {elapsed_time:.2f} seconds')
         return embedding
    
    def create_tables(self)->None:
        self.vs_client.create_tables()
    
    def create_index(self)->None:
        self.vs_client.create_embedding_index(client.DiskAnnIndex())

    def drop_index(self)->None:
        self.vs_client.drop_index()

    def upsert(self, df:pd.DataFrame)->None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.vs_client.upsert(list(records))
        ai_logger.info(f"Inserted {len(df)} records into {self.vector_settings.table_name}")






if __name__ == '__main__':
    __all__=["VectorStore"]

         


        
    
