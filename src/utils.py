import json
from typing import List, Dict, Any
from dataclasses import dataclass, field
import pickle

# TODO: fix the default value
@dataclass
class Result:
    topic: Dict
    subquestions: List[str]
    hits: List[Dict] = None 
    evidences: Dict = None
    pools: Dict = None
    clusters: List[Dict] = None
    responses: List = field(default_factory=list)
    citations: List[Dict] = None

def convert_result_to_submission(
    outputs: List[Result],
    filename: str = "submission.jsonl",
    team_id: str = "hltcoe",
    task: str = "English",
    run_id: str = "decontext-v0",
):
    with open(filename, 'w') as f:

        for output in outputs:
            if len(output.responses) == 0:
                print(output.responses)
                raise ValueError("All outputs must be instances of the Result class.")

            # citations = [list(r['citations'].keys()) for r in output.responses]
            # citations = list(set(sum(citations, [])))
            submission = {
                "metadata": {
                    "run_id": run_id,
                    "topic_id": output.topic['request_id'],
                    "team_id": team_id,
                    "task": task,
                },
                "responses": output.responses,
                "references": []
            }

            f.write(json.dumps(submission, ensure_ascii=False)+'\n')

def save(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load(filename: str):
    results = pickle.load(open(filename, 'rb'))
    return results
