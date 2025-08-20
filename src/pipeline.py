import pickle
import json
import pre_retrieval
import retrieval
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
# version = 'v2.1'
# n_questions = 10
# k = 2000
# k = 5000 if version.endswith('.2') else k
# rel_threshold = 0.95

##### v3
version = 'v3'
n_questions = 20
k = 2000
rel_threshold = 0.9

##### shared hyperparameters
n_centroids = 10
n_neighbors = 20  
sim_threshold = 0.9
encode_type = 'allsep'
encode_type = 'instruct-' + encode_type
agg_type = 'reflect'
max_char_limit = 2000
discard_limit = 5

# ### Hyperparameters ###

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
# temp_results = retrieval.run(
#         temp_results,
#         index='/exp/scale25/artifacts/decontextualized_corpus/neuclir/index/flatten.mlir.mt.lucene',
#         k=k,
# )
# results[2] = temp_results
# print(f"Number of evidences: {len(temp_results[0].evidences)}")
# save(results, '/exp/jhueiju/temp2.pkl')

# 3. Retrieval - filter
# results = pickle.load(open('/exp/jhueiju/temp2.pkl', 'rb'))
# temp_results = filtering.run(
#         results[2],
#         corpus_path='/exp/scale25/neuclir/docs/mlir.mt.jsonl',
#         rel_threshold=rel_threshold,
#         top_k=999999,
# )
# results[3] = temp_results
# print(f"Number of evidences after filtering: {len(temp_results[0].evidences)}")
# save(results, '/exp/jhueiju/temp3.pkl')

# 4. Post retrieval - Clustering
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
results = pickle.load(open('/exp/jhueiju/temp4.pkl', 'rb'))
if 'dfs' in version:
    import generate_reflection_dfs as generate_reflection
else:
    import generate_reflection_bfs as generate_reflection

temp_results = generate_reflection.run(results[4], limit=discard_limit, max_char_limit=max_char_limit, use_litellm=('qwen3' not in version))
results[5] = temp_results
save(results, '/exp/jhueiju/temp5.pkl')

# 6. Refinement 
# results = pickle.load(open('/exp/jhueiju/temp5.pkl', 'rb'))
# temp_results = copy.deepcopy(results[5])
# import generate_refine
# temp_results = generate_refine.run(
#     results[5],
#     max_char_limit=max_char_limit
# )

# N: Final stage: Convert results to submission format
convert_result_to_submission(
    outputs=temp_results,
    filename=f"../submissions/decontext-{version}.{encode_type}.rel{rel_threshold}.c{n_centroids}.k{n_neighbors}.agg{agg_type}.jsonl",
    team_id="hltcoe",
    task="English",
    run_id=f"decontext-{version}.{encode_type}.rel{rel_threshold}.c{n_centroids}.k{n_neighbors}.agg{agg_type}",
)
print("Pipeline completed successfully\nThe configurations:\n"
        f"version: {version}, n_questions: {n_questions}, k: {k}, rel_threshold: {rel_threshold}, "
        f"n_centroids: {n_centroids}, n_neighbors: {n_neighbors}, sim_threshold: {sim_threshold}, "
        f"encode_type: {encode_type}, agg_type: {agg_type}.\n"
        "Results saved to submission file.")
