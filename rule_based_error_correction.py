"""Rule-based error correction for SQL queries
"""

import re
import json
from collections import defaultdict
import argparse
import os
from pathlib import Path


from dataset_classes.spider_dataset import SpiderDataset, BirdDataset
from utils.dataset_utils import load_line_by_line_json, save_line_by_line_json, load_txt_records


# Regular expression to find escaped double quoted strings
# pattern = r'\\"(.*?)\\"'
# pattern = r'"(.*?)"'
pattern = r'(?:(?:\\)?[\"\'])(.*?)(?:(?:\\)?[\"\'])'


def load_content_index_file(input_file_path):
    """split_name -> db_id -> table_name -> column_name -> set of values"""
    with open(input_file_path, 'r') as input_file:
        content_index = json.load(input_file)
    for split_name in content_index.keys():
        for db_id in content_index[split_name].keys():
            all_vals = set()
            for table_name in content_index[split_name][db_id].keys():
                for column_name in content_index[split_name][db_id][table_name].keys():
                    all_vals.update(set(content_index[split_name][db_id][table_name][column_name]))
            content_index[split_name][db_id] = {
                'groups': convert_str_list_to_groups(list(all_vals)),
                'vals': all_vals,
            }            
    return content_index

def generate_val_set(content_index):
    """Convert all values in each db_id into a single set"""
    db2vals = dict()
    for split_name in content_index.keys():
        db2vals[split_name] = dict()
        for db_id in content_index[split_name].keys():
            all_vals = set()
            for table_name in content_index[split_name][db_id].keys():
                for column_name in content_index[split_name][db_id][table_name].keys():
                    all_vals.update(content_index[split_name][db_id][table_name][column_name])
            db2vals[split_name][db_id] = all_vals
    return db2vals


# Function to use with re.sub to replace each match
def replace_func(match, proc_func=None, db_groups=None, db_vals=None):
    original_string = match.group(1)  # The captured group is the content within the quotes
    processed_string = proc_func(original_string, db_groups=db_groups, db_vals=db_vals)  # Process the string
    flag_changed = processed_string != original_string
    return flag_changed, "\'" + processed_string + "\'"


def convert_str_list_to_groups(str_list):
    """Convert values in a string list into groups based on the processed string, skip non-string values"""
    group2variants = defaultdict(set)
    for input_str in str_list:
        if isinstance(input_str, str):
            processed_str = input_str.lower().strip()
            group2variants[processed_str].add(input_str)
    return group2variants

def get_variants(input_str, group2variants):
    processed_str = input_str.lower().strip()
    if processed_str not in group2variants:
        return None
    return group2variants[processed_str]

def proc_str(input_str, db_groups, db_vals):
    if not db_groups or not db_vals or input_str in db_vals:
        return input_str
    variants = get_variants(input_str, db_groups)
    if not variants:
        ## input string is not in the string list
        return input_str
    else:
        return next(iter(variants))


def process_query_by_db_values(query, proc_func, db_groups, db_vals):
    flag_changed = False  # Initialize flag

    def replacement(match):
        nonlocal flag_changed  # Access the nonlocal flag_changed variable
        change, new_string = replace_func(match, proc_func, db_groups, db_vals)
        if change:
            flag_changed = True  # Set flag if there's a change
        return new_string

    processed_query = re.sub(pattern, replacement, query)  # Replace all matches
    return flag_changed, processed_query


def main(args):
    ## content index file stores the db content index if you want to use db values for error correction
    ## example generation script could be found at utils.generate_content_index.py
    try:
        content_index = load_content_index_file(args.content_index_file_path)
        print(f"Loaded content index from {args.content_index_file_path}")
    except:
        content_index = {}
    
    #### Step 1: load the dataset
    ## reason: need database id and table.json for self-correction prompt generation
    dataset_name = args.dataset_name
    dataset_dir_path = args.dataset_dir_path
    ## if the dataset_dir_path is not provided, use the default path datasets/{dataset_name}
    if not dataset_dir_path:
        dataset_dir_path = os.path.join(os.path.dirname(__file__), 'datasets', dataset_name)
    if args.dataset_name == 'spider':
        dataset = SpiderDataset(dataset_dir_path)
    if args.dataset_name == 'bird':
        dataset = BirdDataset(dataset_dir_path)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset_name}")
    if args.split_name == 'dev':
        split_name = 'dev'
        table_schema_path = dataset.dev_table_schema_path
    elif args.split_name == 'test':
        table_schema_path = dataset.test_table_schema_path
        split_name = 'test'
    else:
        raise ValueError(f"Invalid split name {args.split_name}")
    db_id_list = [x['db_id'] for x in dataset.data[split_name]]
    questions = [x['question'] for x in dataset.data[split_name]]
    if args.first_n > 0:
        db_id_list = db_id_list[:args.first_n]
        questions = questions[:args.first_n]
    print(f"Loaded {len(db_id_list)} db_ids and {len(questions)} questions from {dataset_dir_path}")

    #### Step 2: load the generated prompt results
    records = load_txt_records(args.prev_response_file_path)
    assert len(records) == len(dataset.data[split_name]), f"Number of previous prompt results {len(records)} does not match number of records {len(dataset.data[split_name])}"
    print(f"Loaded {len(records)} previous prompt results from {args.prev_response_file_path}")

    #### Step 3: fix prompts by rule-based error correction
    corrected_sql_list = []
    changed_idx_list = []
    for query, data_dict in zip(records, dataset.data[split_name]):
        db_id = data_dict['db_id']
        db_groups = content_index[split_name][db_id]['groups']
        db_vals = content_index[split_name][db_id]['vals']
        flag_changed, processed_query = process_query_by_db_values(query, proc_str, db_groups, db_vals)
        if flag_changed:
            print(f"Changed query for idx {data_dict['idx']}")
            print(f"Question:\n{data_dict['question']}")
            print(f"Gold query:\n{data_dict['query']}")
            print(f"Original query:\n{query}")
            print(f"Processed query:\n{processed_query}")
            print("\n")
            changed_idx_list.append(data_dict['idx'])
        corrected_sql_list.append(processed_query)
    ## save to file
    output_file_path = args.response_file_path
    Path.mkdir(Path(output_file_path).parent, parents=True, exist_ok=True)
    with open(output_file_path, 'w') as output_file:
        for corrected_sql in corrected_sql_list:
            output_file.write(corrected_sql + '\n')
    print(f"Saved {len(corrected_sql_list)} corrected prompts to {output_file_path}")
    print(f"Changed idxs: {changed_idx_list}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Correct Errors in Generated Prompts')
    parser.add_argument('--dataset_name', type=str, help='the name of dataset', default='spider', choices=['spider', 'bird'])
    parser.add_argument('--dataset_dir_path', type=str, help='the directory of dataset', default='')
    parser.add_argument('--split_name', type=str, help='the name of split', default='test', choices=['test', 'dev'])
    parser.add_argument('--content_index_file_path', type=str, help='file path that store db content index', default='')
    parser.add_argument('--prev_response_file_path', type=str, help='file path to store the previous prompt results', default='')
    parser.add_argument('--response_file_path', type=str, help='file path to store the prompt results', default='')
    parser.add_argument("--first_n", type=int, help="only run first n records", default=-1)

    main(parser.parse_args())

