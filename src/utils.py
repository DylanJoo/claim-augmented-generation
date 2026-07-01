import json
import pickle
from typing import List, Dict, Any
from dataclasses import dataclass, field

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

class Hit(dict):
    def __init__(self, docid: str, score: float, rank: int = -1, content_dict: dict = None):
        super().__init__(docid=docid, score=score, rank=rank, content_dict=content_dict or {})

    @property
    def docid(self) -> str: return self["docid"]
    @property
    def score(self) -> float: return self["score"]
    @property
    def rank(self) -> int: return self["rank"]
    @property
    def content_dict(self) -> str: return self["content_dict"]
    @property
    def content(self) -> str: return self["content_dict"]["text"]
    @property
    def title(self) -> str: return self["content_dict"]["title"]

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

def load_topics(path: str) -> List[Dict]:
    topics = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                topics.append(json.loads(line))
    return topics


def save(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load(filename: str):
    results = pickle.load(open(filename, 'rb'))
    return results
