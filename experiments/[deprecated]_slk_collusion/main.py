from tiny_eval.utils.jsonl import read_jsonl
from pydantic import BaseModel

class TaskEntry(BaseModel):
    original_ix: int
    year_2023: bool = False  # Using year_2023 since '2023' is not a valid variable name
    task: str
    code_template: str
    original_code: str
    code: str
    task_looks_malicious: int
    compile: int
    unsafe: int
    looks_unsafe: int
    security_related: int

# def select_unsafe_train_entry():
#     """
#     Select an unsafe train entry and save it to a file.
#     """

#     data = read_jsonl('data/unsafe_train.jsonl')
#     # Parse the first entry as a TaskEntry
#     for i, entry in enumerate(data[:100]):
#         entry = TaskEntry(**entry)

#         if i == 97:
#             print("-" * 50)
#             print(i)
#             print(entry.code)
#             # save the entry
#             with open(f'data/unsafe_train_{i}.json', 'w') as f:
#                 f.write(entry.model_dump_json())
#             break

def viz_entry():
    import json
    import pathlib 

    curr_dir = pathlib.Path(__file__).parent
    with open(curr_dir / 'data' / 'unsafe_train_97_my-chatbot.json', 'r') as f:
        data = json.load(f)
    entry = TaskEntry(**data)
    print(entry.code)

if __name__ == '__main__':
    viz_entry()