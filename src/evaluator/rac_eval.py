import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
import json
import argparse
import numpy as np
from collections import defaultdict
import ir_measures
from ir_measures import Metric, MAP, nDCG, P, alpha_nDCG, StRecall
import pandas as pd


def load_run_or_qrel(path, topk=1000, threshold=1):
    run_dict = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            try:
                qid, _, docid, rank, score, _ = line.strip().split()
                if int(rank) <= topk:
                    run_dict[qid][docid] = float(score)
            except ValueError:
                qid, iteration, docid, rel = line.strip().split()
                if int(rel) >= threshold:
                    run_dict[qid][docid] = int(rel)
    return run_dict


def load_diversity_qrel(path):
    df = pd.read_csv(path, sep=r'\s+', names=['query_id', 'iteration', 'doc_id', 'relevance'])
    df['query_id'] = df['query_id'].astype(str)
    df['doc_id'] = df['doc_id'].astype(str)
    return df


def load_ratings(path):
    ratings = defaultdict(lambda: defaultdict(lambda: None))
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            ratings[str(data['id'])][str(data['docid'])] = data['rating']
    return ratings


def coverage_measures(ratings, ratings_oracle, filter_by_oracle=False, tau=3):
    if filter_by_oracle:
        answerable = np.array([r >= tau for r in ratings_oracle])
    else:
        answerable = np.array([True for _ in ratings])
    value = sum(ratings[answerable] >= tau) / sum(answerable)
    return Metric(query_id='dummy', value=value, measure='Cov')



def rac_eval(run, qrel, div_qrel, judge, tau=3, filter_by_oracle=False):
    outputs = defaultdict(list)

    for metric in ir_measures.iter_calc([alpha_nDCG@10, alpha_nDCG@20, StRecall@10, StRecall@20], div_qrel, run):
        outputs[str(metric.measure)].append(metric.value)

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--qrel", type=str, required=True)
    parser.add_argument("--judge", type=str, required=True,
                        help="jsonl: {'id': str, 'docid': str, 'rating': List[int]}")
    parser.add_argument("--filter_by_oracle", action="store_true", default=False)
    parser.add_argument("--tau", type=int, default=3)
    args = parser.parse_args()

    run = load_run_or_qrel(args.run, topk=1000)
    qrel = load_run_or_qrel(args.qrel, threshold=1)
    div_qrel = load_diversity_qrel(args.qrel)
    ratings = load_ratings(args.judge)

    missing_qids = [qid for qid in qrel if qid not in run]
    if missing_qids:
        qrel = {k: v for k, v in qrel.items() if k in run}
        div_qrel = div_qrel[div_qrel['query_id'].isin(run.keys())]
        logger.warning(f"Missing results for {len(missing_qids)} topics; evaluating on {len(qrel)}")

    outputs = rac_eval(
        run=run,
        qrel=qrel,
        div_qrel=div_qrel,
        judge=ratings,
        tau=args.tau,
        filter_by_oracle=args.filter_by_oracle,
    )

    sys.stdout.write(args.run.rsplit('/', 1)[0] + " | " + args.run.rsplit('/')[-1] + " | ")

    for key, values in sorted(outputs.items()):
        sys.stdout.write(key + " | ")
        sys.stdout.write("{:.4f}".format(np.mean(values)) + " | ")
    sys.stdout.write("\n")
