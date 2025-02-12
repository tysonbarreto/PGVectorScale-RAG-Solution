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

    def search(self,
               query_text:str,
               limit:int=5,
               metadata_filter: Union[dict,list[dict]]=None,
               predicates: Optional[client.Predicates] = None,
               time_range: Optional[Tuple[datetime,datetime]]=None,
               return_dataframe:bool = True)->Union[List[Tuple[Any,...]]]:
        """
        Query the vector database for similar embeddings based on input text.

        More info:
            https://github.com/timescale/docs/blob/latest/ai/python-interface-for-pgvector-and-timescale-vector.md

        Args:
            query_text: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary or list of dictionaries for equality-based metadata filtering.
            predicates: A Predicates object for complex metadata filtering.
                - Predicates objects are defined by the name of the metadata key, an operator, and a value.
                - Operators: ==, !=, >, >=, <, <=
                - & is used to combine multiple predicates with AND operator.
                - | is used to combine multiple predicates with OR operator.
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.

        Basic Examples:
            Basic search:
                vector_store.search("What are your shipping options?")
            Search with metadata filter:
                vector_store.search("Shipping options", metadata_filter={"category": "Shipping"})
        
        Predicates Examples:
            Search with predicates:
                vector_store.search("Pricing", predicates=client.Predicates("price", ">", 100))
            Search with complex combined predicates:
                complex_pred = (client.Predicates("category", "==", "Electronics") & client.Predicates("price", "<", 1000)) | \
                               (client.Predicates("category", "==", "Books") & client.Predicates("rating", ">=", 4.5))
                vector_store.search("High-quality products", predicates=complex_pred)
        
        Time-based filtering:
            Search with time range:
                vector_store.search("Recent updates", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)))
        """

            #    query_text:str,
            #    limit:int=5,
            #    metadata_filter: Union[dict,list[dict]]=None,
            #    predicates: Optional[client.Predicates] = None,
            #    time_range: Optional[Tuple[datetime,datetime]]=None,
            #    return_dataframe:bool = True
        
        query_embedding = self.get_embedding(query_text)
        start_time = time.time()
        search_args = {"limit": limit}
        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = self.vec_client.search(query_embedding, **search_args)
            
        elapsed_time = time.time() - start_time

        ai_logger.info(f"Vector search completed in {elapsed_time:.3f} seconds")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
        """
        # Convert results to DataFrame
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )

        # Expand metadata column
        df = pd.concat(
            [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
        )

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)

        return df
    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.

        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.

        Examples:
            Delete by IDs:
                vector_store.delete(ids=["8ab544ae-766a-11ef-81cb-decf757b836d"])

            Delete by metadata filter:
                vector_store.delete(metadata_filter={"category": "Shipping"})

            Delete all records:
                vector_store.delete(delete_all=True)
        """
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )

        if delete_all:
            self.vec_client.delete_all()
            ai_logger.info(f"Deleted all records from {self.vector_settings.table_name}")
        elif ids:
            self.vec_client.delete_by_ids(ids)
            ai_logger.info(
                f"Deleted {len(ids)} records from {self.vector_settings.table_name}"
            )
        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            ai_logger.info(
                f"Deleted records matching metadata filter from {self.vector_settings.table_name}"
            )

if __name__ == '__main__':
    __all__=["VectorStore"]

         


        
    
