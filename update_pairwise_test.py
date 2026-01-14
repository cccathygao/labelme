import json
import argparse

def update_pairwise_test(input_file, output_file=None):
    """
    Update pairwise_test.json by appending choices to the question field.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, overwrites input file)
    """
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Append string to each entry's question
    choices_string = ". Choices: A. The first image. B. The second image."
    
    updated_count = 0
    for entry in data:
        if 'question' in entry:
            # Only append if not already present
            if not entry['question'].endswith(choices_string):
                entry['question'] += choices_string
                updated_count += 1
        if 'prompt' in entry:
            # Only append if not already present
            if not entry['prompt'].endswith(choices_string):
                entry['prompt'] += choices_string
                updated_count += 1
    
    # Write back to file
    if output_file is None:
        output_file = input_file
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated {updated_count} entries in {output_file}")
    print(f"Total entries: {len(data)}")

def main():
    parser = argparse.ArgumentParser(description="Update pairwise_test.json with choices in question field")
    parser.add_argument("--input", default="test_bed/pairwise_test.json", help="Input JSON file path")
    parser.add_argument("--output", default=None, help="Output JSON file path (default: overwrite input)")
    
    args = parser.parse_args()
    
    update_pairwise_test(args.input, args.output)

if __name__ == "__main__":
    main()