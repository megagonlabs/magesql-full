conda activate <placeholder_conda_env>
PROJECT_PATH=<placeholder_repo_path>
cd $PROJECT_PATH
export PYTHONPATH=$PYTHONPATH:`pwd`

python -c "import nltk; nltk.download('punkt_tab')"

# Run text2sql generation with Jaccard demonstration selector on Spider dev split.
python construct_prompt.py --dataset_name spider --demonstration_selector jaccard --num_demonstrations 2 --seed 1234 --template_option template_option_1 --split_name dev --prompt_file_path <placeholder_for_prompt_file_path> 

# Run prompting with GPT-4 to get initial responses
python prompt_llm.py --prompt_file_path <placeholder_for_prompt_file_path>  --response_file_path <placeholder_for_response_file_path>  --first_n -1 --openai_api_key <placeholder_openai_api_key> --openai_organization <placeholder_openai_organization> --model gpt-4

# Run prompting based error correction
python error_correction.py --prev_response_file_path <placeholder_for_previous_response_file_path> --prompt_file_path <placeholder_for_error_correction_prompt_file_path> --response_file_path <placeholder_for_error_correction_response_file_path> --first_n -1 --openai_api_key <placeholder_openai_api_key> --openai_organization <placeholder_openai_organization> --model gpt-4 --rules_groups "1 3 4" --split_name dev

# Run rule-based error correction (optional)
python rule_based_error_correction.py --dataset_name spider --split_name dev --content_index_file_path <placeholder_content_index_file_path> --prev_response_file_path <placeholder_for_previous_response_file_path> --response_file_path <placeholder_for_rule_based_error_correction_response_file_path>
