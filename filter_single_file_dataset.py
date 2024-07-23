import gzip
import sys
import json

if __name__ == "__main__":
    path = sys.argv[1]
    validation_path = sys.argv[2]

    validation_data = []
    with open(validation_path, 'rt') as f:
        for line in f:
            validation_data.append(line.replace("\n", ""))

        print(f"Validation topics: {len(validation_data)}")

    with gzip.open(path, 'rt') as f, gzip.open(path.replace("intermediate_dataset/", "export/"), 'wt') as f_out:
        for line in f:
            if line.strip():
                if json.loads(line)['title'] in validation_data:
                    print(f"match of '{json.loads(line)['title']}'")
                else:
                    f_out.write(line)
                    #f_out.write("\n")
