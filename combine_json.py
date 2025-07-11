import json

# Read the first JSON file
with open('datasets/spider/train_spider.json', 'r') as f:
    spider_data = json.load(f)

# Read the second JSON file
with open('datasets/spider/train_others.json', 'r') as f:
    others_data = json.load(f)

# Combine the data
combined_data = spider_data + others_data

# Write the combined data to a new file
with open('datasets/spider/train_spider_and_others.json', 'w') as f:
    json.dump(combined_data, f, indent=2)

print(f"Combined {len(spider_data)} entries from train_spider.json")
print(f"Combined {len(others_data)} entries from train_others.json")
print(f"Total entries in combined file: {len(combined_data)}") 