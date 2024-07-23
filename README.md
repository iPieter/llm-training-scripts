
## Data preparation

```bash
python prepare_dataset.py 2020 # and other years that you want to include
python create_hf_dataset.py 
ls intermediate_dataset/*_train.jsonl.gz | ~/bin/usr/bin/parallel --jobs 30 --progress python filter_single_file_dataset.py {} notebooks/validation_articles.txt
 ```