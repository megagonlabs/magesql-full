"""
This component aims at getting respone for all prompts in a file
"""
import argparse
import os
import json
from typing import List
from pathlib import Path
import time

from tqdm import tqdm
import openai

from utils.openai_utils import init_gpt, init_openai_client, get_prompt_from_openai
from utils.dataset_utils import load_line_by_line_json, save_line_by_line_json
from utils.sql_str_utils import query_postprocessing


def main(args):
    #### Step 1: load the generated prompts
    prompts = load_line_by_line_json(args.prompt_file_path)
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file_path}")
    questions = [prompt['prompt'] for prompt in prompts]
    assert len(questions) > 0
    if args.first_n > 0:
        questions = questions[:args.first_n]
        print(f"Only run first {args.first_n} prompts")

    #### Step 2: generate response for each prompt
    ## init openai api
    init_gpt(args.openai_api_key, args.openai_organization)

    token_cnt = 0
    output_path = args.response_file_path
    if not output_path:
        ## save in directory ../responses/{prompt_file_name} from args.prompt_file_path
        output_file_name = os.path.splitext(os.path.basename(args.prompt_file_path))[0] + '.txt'
        output_path = os.path.join(os.path.dirname(os.path.dirname(args.prompt_file_path)), 'responses', output_file_name)

    Path.mkdir(Path(output_path).parent, parents=True, exist_ok=True)

    start_time = time.time()
    print(f"Generating responses for {len(questions)} prompts")
    ## initialize openai client
    client = init_openai_client(args.openai_api_key, args.openai_organization)
    with open(output_path, 'w') as f:
        for i, question in enumerate(tqdm(questions)):
            res = get_prompt_from_openai(client, args.model, question, args.temperature, args.n)
            if res is None:
                print(f"Failed to generate response for question {i}")
                f.write("\n")
                continue
            # parse result
            token_cnt += res["total_tokens"]
            if args.n == 1:
                for sql in res["response"]:
                    sql = query_postprocessing(sql)
                    f.write(sql)
            else:
                ## TODO: handle multiple sqls
                pass
    end_time = time.time()
    print(f"Total time: {end_time - start_time}, time per question: {(end_time - start_time) / len(questions)}")
    print(f"Total tokens in results: {token_cnt}")
    print(f"Average tokens per question: {token_cnt / len(questions)}")
    print(f"Results saved to {output_path}")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate phone-screen edges between jobpostings and resumes')
    parser.add_argument('--prompt_file_path', type=str, help='file path to store the prompts', default='')
    parser.add_argument('--response_file_path', type=str, help='file path to store the prompt results', default='')
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--openai_organization", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n", type=int, default=1, help="# of sqls to generate for each question")
    parser.add_argument("--first_n", type=int, help="only run first n records", default=100)
    main(parser.parse_args())