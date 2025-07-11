"""Build index for table content"""
import argparse
import os
import json
from pathlib import Path

import sqlite3
from sqlite3 import OperationalError

from dataset_classes.spider_dataset import SpiderDataset, BirdDataset

def build_index_for_tables():
    pass


def get_all_table_contents(db_file_path):
    import sqlite3
    pass


def safe_decode(text, encodings=('utf-8', 'windows-1254', 'iso-8859-9')):
    for enc in encodings:
        try:
            return text.decode(enc)
        except UnicodeDecodeError:
            continue
    return text  # Return as is if all decodings fail

def get_distinct_table_contents(db_path):
    # Connect to the SQLite database
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Dictionary to hold distinct data from all tables
    all_tables_data = {}
    
    # Query each table and fetch the distinct contents of each column
    for table_name in tables:
        table = table_name[0]
        # print(f"Distinct contents of table {table}:")
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        # Dictionary to store distinct values by column for the current table
        table_data = {}
        
        for column in columns:
            column_name = column[1]
            cursor.execute(f'SELECT DISTINCT "{column_name}" FROM "{table}"')
            try:
                distinct_values = cursor.fetchall()
            except OperationalError as e:
                ## if there is error in data fetching, skip it
                print(f"Error in fetching data from table {table} column {column_name} in db {db_path}: {e}")
                continue
            
            # Apply encoding conversion to each distinct value
            distinct_values = [safe_decode(value[0]) if isinstance(value[0], bytes) else value[0] for value in distinct_values]
            # distinct_values = [value[0] for value in distinct_values]  # Flatten list of tuples
            table_data[column_name] = distinct_values
            # print(f"Column {column_name}: {distinct_values}")
        
        all_tables_data[table] = table_data
        # print("\n")  # Print a newline for better separation in output

    # Close the connection
    cursor.close()
    connection.close()
    
    return all_tables_data

def main(args):
    #### Step 1: load the dataset
    dataset_name = args.dataset_name
    dataset_dir_path = args.dataset_dir_path
    ## if the dataset_dir_path is not provided, use the default path datasets/{dataset_name}
    if not dataset_dir_path:
        dataset_dir_path = os.path.join(os.path.dirname(__file__), 'datasets', dataset_name)
    if args.dataset_name == 'spider':
        dataset = SpiderDataset(dataset_dir_path)
    elif args.dataset_name == 'bird':
        dataset = BirdDataset(dataset_dir_path)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset_name}")   

    #### Step 2: build index for table content
    ## get all table paths
    dataset_data = dict()
    db_paths = dataset.db_paths
    for split_name in dataset.data.keys():
        dataset_data[split_name] = dict()
        for db_id in db_paths[split_name].keys():
            db_path = db_paths[split_name][db_id]
            all_tables_data = get_distinct_table_contents(db_path)
            dataset_data[split_name][db_id] = all_tables_data

    #### Step 3: save the table content index
    output_file_name = os.path.join(args.dataset_dir_path, 'content_index', args.output_file_name)
    Path.mkdir(Path(output_file_name).parent, parents=True, exist_ok=True)
    with open(output_file_name, "w") as outfile: 
        json.dump(dataset_data, outfile)
    print(f"Saved table content index to {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Correct Errors in Generated Prompts')
    parser.add_argument('--dataset_name', type=str, help='the name of dataset', default='spider', choices=['spider', 'bird'])
    parser.add_argument('--dataset_dir_path', type=str, help='the directory of dataset', default='')
    # parser.add_argument('--split_name', type=str, help='the name of split', default='test', choices=['test', 'dev'])
    parser.add_argument('--output_file_name', type=str, help='output file path that store db content index', default='')
    main(parser.parse_args())