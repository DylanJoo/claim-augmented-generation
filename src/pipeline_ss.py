import pickle
import json
import pre_retrieval
import retrieval_ss
import filtering
import post_retrieval
import copy
from utils import convert_result_to_submission, save, load

## Start the pipeline
results = {}

### Hyperparameters ###
##### v0
# version = 'v0'
# n_questions = 10
# k = 2000
# rel_threshold = 0.7

##### v1
# version = 'v1'
# n_questions = 10
# k = 2000
# rel_threshold = 0.9
# agg_type = ''

##### v2
version = 'v2.ss.1'
n_questions = 10
k = 5000 if version.endswith('.1') else 2000
k = 10000 if version.endswith('.2') else k
rel_threshold = 0.9

##### shared hyperparameters
agg_type = 'reflect'
n_centroids = 20
n_neighbors = 10  
encode_type = 'allsep' # sep allsep, addadjacent
encode_type = 'instruct' + encode_type
sim_threshold = 0.99 if 'instruct' in encode_type else 0.9
max_char_limit = 10000 if (version.endswith('.1') or version.endswith('.2') ) else 2000
### Hyperparameters ###

# # 0. Load topics
# with open("/exp/scale25/neuclir/topics/neuclir24-test-request.jsonl", "r") as f:
#     topics = [json.loads(line) for line in f]
#
# # 1. Pre-retrieval
# temp_results = pre_retrieval.run(
#         topics, 
#         n_questions=n_questions,
# )
# results[1] = temp_results
# print(f"Number of subquestions: {len(temp_results[0].subquestions)}")
# save(results, '/exp/jhueiju/temp1.pkl')
#
# # 2. Retrieval
# results = pickle.load(open('/exp/jhueiju/temp1.pkl', 'rb'))
# temp_results = retrieval_ss.run(
#         results[1],
#         index='/exp/scale25/artifacts/decontextualized_corpus/neuclir/index/flatten.mlir.mt.lucene',
#         k=k,
# )
# results[2] = temp_results
# print(f"Number of evidences: {len(temp_results[0].evidences)}")
# save(results, '/exp/jhueiju/temp2.pkl')
#
# # 3. Retrieval - filter
# results = pickle.load(open('/exp/jhueiju/temp2.pkl', 'rb'))
# temp_results = filtering.run(
#         results[2],
#         corpus_path='/exp/scale25/neuclir/docs/mlir.mt.jsonl',
#         rel_threshold=rel_threshold,
# )
# results[3] = temp_results
# print(f"Number of evidences after filtering: {len(temp_results[0].evidences)}")
# save(results, '/exp/jhueiju/temp3.pkl')
#
# # 4. Post retrieval - Clustering
# results = pickle.load(open('/exp/jhueiju/temp3.pkl', 'rb'))
# temp_results = post_retrieval.run(
#         results[3],
#         n_centroids=n_centroids,
#         n_neighbors=max(n_neighbors, 2),  # at least 2 otherwise indexing error causes
#         sim_threshold=sim_threshold,
#         add_all=('allsep' in encode_type),
#         concat_adjacent=('addadjacent' in encode_type),
#         add_condition=('instruct' in encode_type),
#         verbose=False
# )
# results[4] = temp_results
# print(f"Number of clusters: {len(temp_results[0].clusters)}")
# save(results, '/exp/jhueiju/temp4.pkl')

# 5. Cluster generation (temp.pkl)
results = pickle.load(open('/exp/jhueiju/temp4-2.pkl', 'rb'))
import generate_reflection
temp_results = generate_reflection.run(results[4], max_char_limit=max_char_limit, use_litellm=True)
results[5] = temp_results

save(results, '/exp/jhueiju/temp5.pkl')

# N: Final stage: Convert results to submission format
convert_result_to_submission(
    outputs=temp_results,
    filename=f"../submissions/decontext-{version}.{encode_type}.c{n_centroids}.k{n_neighbors}.agg{agg_type}.jsonl",
    team_id="hltcoe",
    task="English",
    run_id=f"decontext-{version}.{encode_type}.c{n_centroids}.k{n_neighbors}.agg{agg_type}",
)
print("Pipeline completed successfully\nThe configurations:\n"
        f"version: {version}, n_questions: {n_questions}, k: {k}, rel_threshold: {rel_threshold}, "
        f"n_centroids: {n_centroids}, n_neighbors: {n_neighbors}, sim_threshold: {sim_threshold}, "
        f"encode_type: {encode_type}, agg_type: {agg_type}.\n"
        "Results saved to submission file.")
