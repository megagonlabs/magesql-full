"""
This component aims at generating a prompt for a given input NL question.
The prompt consists of 3 parts: the instruction, the demonstration and the question.
"""
import argparse
import os
import json
from typing import List
from pathlib import Path

from tqdm import tqdm

from dataset_classes.spider_dataset import SpiderDataset
from dataset_classes.bird_dataset import BirdDataset
from demonstration_selector.first_k_demonstration_selector import FirstKDemonstrationSelector
from demonstration_selector.random_demonstration_selector import RandomDemonstrationSelector
from demonstration_selector.hardness_demonstration_selector import HardnessDemonstrationSelector
from demonstration_selector.jac_demonstration_selector import JacDemonstrationSelector
from demonstration_selector.struct_demonstration_selector import StructDemonstrationSelector
from demonstration_selector.gcl_demonstration_selector import GCLDemonstrationSelector
from utils.template_utils import get_template, fill_template
from utils.openai_utils import get_price_from_tokens

def save_prompts(data:List[dict], output_path:str):
    Path.mkdir(Path(output_path).parent, parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')

def main(args):
    #### Step 1: load the dataset
    dataset_name = args.dataset_name
    dataset_dir_path = args.dataset_dir_path
    ## if the dataset_dir_path is not provided, use the default path datasets/{dataset_name}
    if not dataset_dir_path:
        dataset_dir_path = os.path.join(os.path.dirname(__file__), 'datasets', dataset_name)
    if args.dataset_name == 'spider':
        dataset = SpiderDataset(dataset_dir_path, dev_preliminary_queries_file_path=args.dev_preliminary_queries_file_path, test_preliminary_queries_file_path=args.test_preliminary_queries_file_path)
    elif args.dataset_name == 'bird':
        dataset = BirdDataset(dataset_dir_path, dev_preliminary_queries_file_path=args.dev_preliminary_queries_file_path)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset_name}")   

    #### Step 2: select demonstractions for each testing data
    ## intialize demonstration selector
    demonstration_selector = args.demonstration_selector
    if demonstration_selector == 'first_k':
        demonstration_selector = FirstKDemonstrationSelector(dataset)
    elif demonstration_selector == 'random':
        demonstration_selector = RandomDemonstrationSelector(dataset, seed=args.seed)
    elif demonstration_selector == 'hardness':
        demonstration_selector = HardnessDemonstrationSelector(dataset)
    elif demonstration_selector == 'jaccard':
        demonstration_selector = JacDemonstrationSelector(dataset)
    elif demonstration_selector == 'struct':
        demonstration_selector = StructDemonstrationSelector(dataset)
    elif demonstration_selector == 'gcl':
        demonstration_selector = GCLDemonstrationSelector(dataset, encoder_path=args.encoder_path)
    else:
        raise ValueError(f"Invalid demonstration selector: {demonstration_selector}")

    ## Step 3: generate prompt for each testing data
    all_prompts = []
    template = get_template(args.template_option)
    for data in tqdm(dataset.data[args.split_name]):
        demonstrations = demonstration_selector.select_demonstrations(
            data,
            num_demonstrations=args.num_demonstrations,
            flag_return_ids = False
        )
        prompt_generation_input = {
            'idx': data['idx'],
            'question': data['question'],
            'question_toks' : data['question_toks'],
            'query': data['query'],
            'query_toks': data['query_toks'],
            'db_id': data['db_id'],
            'db_path': data['db_path'],
            'demonstrations': demonstrations,
            'demonstrations_idxs': [x['idx'] for x in demonstrations],
            'num_demonstrations': len(demonstrations),
            'evidence': data.get('evidence', None) # only for bird dataset
        }

        provided_schema = None

        template = get_template(args.template_option)
        prompt, num_tokens = fill_template(
            prompt_generation_input, 
            template,
            provided_schema=provided_schema
        )
        prompt_generation_output = {
            'idx': data['idx'],
            'question': data['question'],
            'question_toks' : data['question_toks'],
            'query': data['query'],
            'query_toks': data['query_toks'],
            'db_id': data['db_id'],
            'db_path': data['db_path'],
            # 'demonstrations': demonstrations,
            'demonstrations_idxs': [x['idx'] for x in demonstrations],
            'num_demonstrations': len(demonstrations),
            'prompt': prompt,
            'num_tokens': num_tokens,
            'evidence': data.get('evidence', None) # only for bird dataset
        }
        if num_tokens > 4096:
            print(f"Warning: the prompt for idx {data['idx']} has {num_tokens} tokens, which exceeds the limit of 4096 tokens.")
        all_prompts.append(prompt_generation_output)

    # print total number of tokens
    total_tokens = sum([x['num_tokens'] for x in all_prompts])
    print(f"Total number of tokens: {total_tokens}")
    total_price = get_price_from_tokens(total_tokens, 'gpt-4')
    print(f"Total price if using gpt-4: ${total_price:.2f}")

    ## Step 4: save the prompt to a file
    prompt_file_path = args.prompt_file_path
    if not prompt_file_path:
        prompt_file_path = demonstration_selector.get_default_output_file_path({
            'dataset_name': args.dataset_name,
            'dataset_dir_path': args.dataset_dir_path,
            'num_demonstrations': args.num_demonstrations,
            'seed': args.seed,
            'template_option': args.template_option,
            'split_name': args.split_name,
        })
    Path.mkdir(Path(prompt_file_path).parent, parents=True, exist_ok=True)
    save_prompts(all_prompts, prompt_file_path)
    print(f"Saved prompts to {prompt_file_path}")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_name', type=str, help='the name of dataset', default='spider', choices=['spider', 'bird'])
    parser.add_argument('--dataset_dir_path', type=str, help='the directory of dataset', default='')
    parser.add_argument('--split_name', type=str, help='the name of split', default='test', choices=['test', 'dev'])
    parser.add_argument('--dev_preliminary_queries_file_path', type=str, help='the path of preliminary sql file for dev split', default='<your_dev_preliminary_queries_file_path>')
    parser.add_argument('--test_preliminary_queries_file_path', type=str, help='the path of preliminary sql file for test split', default='<your_test_preliminary_queries_file_path>')
    parser.add_argument('--seed', type=int, help='the random seed', default=1234) 
    parser.add_argument('--num_demonstrations', type=int, help='the number of demonstrations to select', default=5)
    parser.add_argument('--demonstration_selector', type=str, help='the method to choose the demonstrations', default='random', choices=['first_k', 'random', 'hardness', 'jaccard', 'struct', 'gcl'])
    parser.add_argument('--template_option', type=str, help='the option of prompt template', default='template_option_1', choices=['template_option_1', 'template_option_2', 'template_option_3', 'template_option_4'])
    parser.add_argument('--prompt_file_path', type=str, help='file path to store the prompts', default='')
    parser.add_argument('--encoder_path', type=str, help='the path of sql encoder model', default='')
    main(parser.parse_args())