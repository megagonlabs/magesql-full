import os
import json
from abc import abstractmethod

from utils.dataset_utils import load_json_records, load_txt_records
from utils.sql_utils import get_sql_for_database

class BaseDataset(object):
    def __init__(self, dataset_dir_path):
        self.name = self.dataset_name
        self.dataset_dir_path = dataset_dir_path

        self.train_file_path = os.path.join(dataset_dir_path, self.train_file_name)
        self.train_sql_file_path = os.path.join(dataset_dir_path, self.train_sql_file_name)
        self.train_database_dir_path = os.path.join(dataset_dir_path, self.train_database_dir_name)
        self.train_table_schema_path = os.path.join(dataset_dir_path, self.train_table_schema_file_name)

        if self.flag_has_dev:
            self.dev_file_path = os.path.join(dataset_dir_path, self.dev_file_name)
            self.dev_sql_file_path = os.path.join(dataset_dir_path, self.dev_sql_file_name)
            self.dev_database_dir_path = os.path.join(dataset_dir_path, self.dev_database_dir_name)
            self.dev_table_schema_path = os.path.join(dataset_dir_path, self.dev_table_schema_file_name)
        else:
            self.dev_file_path = None
            self.dev_sql_file_path = None
            self.dev_database_dir_path = None
            self.dev_table_schema_path = None

        if self.flag_has_test:
            self.test_file_path = os.path.join(dataset_dir_path, self.test_file_name)
            self.test_sql_file_path = os.path.join(dataset_dir_path, self.test_sql_file_name)
            self.test_database_dir_path = os.path.join(dataset_dir_path, self.test_database_dir_name)
            self.test_table_schema_path = os.path.join(dataset_dir_path, self.test_table_schema_file_name)
        else:
            self.test_file_path = None
            self.test_sql_file_path = None
            self.test_database_dir_path = None
            self.test_table_schema_path = None

        self.data = dict() # store the records of each split
        self.db = dict() # store the databases, not used currently
        self.schema = dict() # store the schema, not used currently
        self.load_data()
    
    @abstractmethod
    def load_data(cls):
        """Load the whole dataset
        """
        raise NotImplementedError

    @abstractmethod
    def tokenize_questions(cls):
        """generate the tokenization results of input NL questions.
        """
        pass
    
    @abstractmethod
    def tokenize_queries(self):
        """generate the tokenization results of input NL SQL.
        """
        pass

    @abstractmethod
    def get_sql_tree(cls):
        """for each SQL in the training set, obtain its parsed format as a labeled tree.
        """
        pass

    @abstractmethod
    def get_sql_sketch(cls):
        """for each NL question, obtain a SQL sketch/clause of it.
        """
        pass

    def _compute_schema(self, split_names=['train', 'dev', 'test']):
        """
        A quick implementation of reusing existing database class to get db schema. Can be optimized further.
        """
        schema = {}
        for split_name in split_names:
            schema[split_name] = {}
            for record in self.data[split_name]:
                db_id = record['db_id']
                db_path = record['db_path'] ## the sqlite db file path
                if db_id not in schema[split_name]:
                    table_schemas = get_sql_for_database(db_path)
                    schema_text = self._convert_to_schema_text(table_schemas)
                    schema[split_name][db_id] = schema_text
            print(f"Schema cache computed for {len(schema[split_name])} databases in {split_name} split.")
        return schema
    
    def _convert_to_schema_text(self, table_schemas, separator:str="\n\n"):
        if not table_schemas:
            return None
        schema_text = separator.join(table_schemas)
        return schema_text