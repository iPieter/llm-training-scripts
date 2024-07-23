from contextlib import ExitStack
import json
import gzip
import random
import time 

GZIP = True
BIN_SIZE = 8000

if __name__ == '__main__':
    years = {
         y :f'data/wikipedia/{y}/enwiki_{y}_clean.jsonl' for y in [2014, 2016, 2018, 2020, 2024]
        }

# we need all files to shuffle them, so let's open them all
with ExitStack() as stack:
    exported_lines = 0
    f_out = (gzip.open(f'export/enwiki_{int(exported_lines / BIN_SIZE)}_train.jsonl.gz', 'at') if GZIP 
                else open(f'export/enwiki_{int(exported_lines / BIN_SIZE)}_train.jsonl', 'a'))
    files = [stack.enter_context(open(fname)) for fname in years.values()]
    start = time.time()
    while len(files) > 0:
        # pick a random file
        f = random.choice(files)
        line = f.readline()
        if not line:
            files.remove(f)
            continue
        if exported_lines % BIN_SIZE == 0:
            end = time.time()
            mins = (end-start)/60
            print(f"Exported {exported_lines/1000}k lines in {mins:.2f} min.", end='\r')
            f_out.close()
            f_out = (gzip.open(f'export/enwiki_{int(exported_lines / BIN_SIZE)}_train.jsonl.gz', 'at') if GZIP 
                else open(f'export/enwiki_{int(exported_lines / BIN_SIZE)}_train.jsonl', 'a'))


        # write the line to the output
        f_out.write(line)
        #f_out.write('\n')
        exported_lines += 1