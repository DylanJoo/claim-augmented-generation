import json
from typing import List
import os
import copy
from tqdm import tqdm 
import re

import pdb

def run(inputs):

    outputs = copy.deepcopy(inputs)

    # Prepare the input to the model
    for i, input in tqdm(enumerate(inputs), desc='Compile cluster', total=len(inputs)):

        # Generate outputs
        collected = []
        for cluster in input.clusters:

            sids = list(cluster.keys())
            docids = list(sid.split('#')[0] for sid in sids)

            for j in range(len(sids)):
                outputs[i].responses.append({"text": cluster[sids[j]], "citations": {docids[j]: 1.0}})
                if sids[j] not in collected:
                    collected.append(sids[j])
                    break

    return outputs  
