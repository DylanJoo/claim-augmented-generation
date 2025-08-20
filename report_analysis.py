import unicodedata
from collections import defaultdict
import argparse
import json
import os

# Example report content format
# {"metadata": {"run_id": "decontext-v0.allsep.c10.k5", "topic_id": "300", "team_id": "hltcoe", "task": "English"}, "responses": [{"text": "Isolation leads to suicidal thoughts", "citations": {"74f8fd2b-ef87-460d-b642-54e499183a9e": 1.0}}, {"text": "Suicide rates in Japan rose sharply in 2020", "citations": {"836170ec-8625-47e6-be91-ff11451617c2": 1.0}}, {"text": "Pandemic exacerbated loneliness in Japan", "citations": {"23435d31-41f2-4af9-b68f-c7fba9542cb8": 1.0}}, {"text": "40% of Japan's suicide prevention organizations closed", "citations": {"c4186ce4-e61f-44d4-9350-ec1e96adaded": 1.0}}, {"text": "Pandemic affects mental health", "citations": {"e5c1ed0a-fa91-4a42-90a4-fa5ad2a91f76": 1.0}}, {"text": "Japan's suicide rate fell 20% in April", "citations": {"c4186ce4-e61f-44d4-9350-ec1e96adaded": 1.0}}, {"text": "Women's employment heavily impacted", "citations": {"fdd59bb9-f872-4399-b23a-cd5b564f6db0": 1.0}}, {"text": "COVID-19 pandemic contributed to rise in Japan's suicide rates", "citations": {"d0614cca-1a44-4531-a44e-315dd1c295d0": 1.0}}, {"text": "Suicide rates rose in 2020", "citations": {"44b5a209-4f36-459e-98f5-5d4777fc3b6e": 1.0}}, {"text": "Japan created a \"minister of loneliness\"", "citations": {"23435d31-41f2-4af9-b68f-c7fba9542cb8": 1.0}}], "references": []}

def analyze_report(report_path):

    if not os.path.exists(report_path):
        print(f"Report file '{report_path}' does not exist.")
        return

    reports = defaultdict(list)
    citations = defaultdict(list)
    with open(report_path, 'r') as file:

        for line in file:
            topic_id = json.loads(line.strip())['metadata']['topic_id']
            responses = json.loads(line.strip())['responses']
            for sentence in responses:
                reports[topic_id] += [sentence['text']]
                citations[topic_id] += list(sentence['citations'].keys())

    stats = defaultdict(list)
    for topic_id in reports:
        report_content = " ".join(reports[topic_id])

        num_length = len(unicodedata.normalize('NFKC', report_content))
        num_sentences = len(reports[topic_id])
        num_citations = len(citations[topic_id]) # TODO: check irrelevant

        print(f"Topic: {topic_id} | Length: {num_length} chars / {num_sentences} sentences")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Analyze a report file.")
    parser.add_argument('report_path', type=str)
    args = parser.parse_args()
    analyze_report(args.report_path)

