from helper import llm_utils
from tqdm import *
from helper.config import EXAMPLES_VERACITY_PATH, K, BATCH_SIZE_VP, SIM_THRESH
from typing import Optional
import time
import json
import os
import argparse

print("K:", K)
print("SIM_THRESH:", SIM_THRESH)
print("EXAMPLES_VERACITY_PATH:", EXAMPLES_VERACITY_PATH)
web_archive_prefix = "https://web.archive.org/web/"

def extract_original_url(url: str) -> str:
    if url.startswith(web_archive_prefix):
        # Find the position after the timestamp (15 characters after the prefix)
        # timestamp_end = len(web_archive_prefix) + 15 => 43
        return url[43:]
    return url

def combine_answers(answers_list):
    processed_answers = []
    ending_punctuation = ".!?;"  # Can add any others like ;

    for ans in answers_list:
        stripped_ans = ans.strip()
        if not stripped_ans:
            continue

        if stripped_ans[-1] not in ending_punctuation:
            processed_answers.append(stripped_ans + ".")
        else:
            processed_answers.append(stripped_ans)
    return " ".join(processed_answers)

def build_examples_content(qv_json):
    prediction_evidence = []
    qv_text = ""
    evidence_map = {}
    for q_id, (q, evidence_list) in enumerate(qv_json.items()):
        qv_text += f"\nQ{q_id+1}: {q}\n"

        ev_id = 1
        final_answers = []
        for evidence_obj in evidence_list:
            # Skip evidence with too less of a similarity score (SIM_THRESH)
            if evidence_obj['score'] < SIM_THRESH:
                continue

            current_evidence = evidence_obj['evidence']
            url = extract_original_url(evidence_obj['source_url'])
            final_answers.append(evidence_obj['evidence'])

            if current_evidence not in evidence_map:
                evidence_map[current_evidence] = f"Q{q_id+1}, A{ev_id}"
                qv_text += f"Q{q_id+1}, A{ev_id}: {current_evidence}\n"
                qv_text += f"Source for Q{q_id+1}, A{ev_id}: {url}\n"
                ev_id += 1
            else:
                qv_text += f"Q{q_id+1}, A{ev_id}: Refer to {evidence_map[current_evidence]}\n"

            # ev_id += 1
            if ev_id > K:
                break

        # Default text, if no evidence was found for that Question
        if not evidence_list or not final_answers:
            qv_text += f"Q{q_id+1}, A1: No evidence.\n"
            final_answers = ["No Evidence."]

        prediction_evidence.append({"question": q, "evidence": combine_answers(final_answers)})
    return qv_text, prediction_evidence

def create_user_task(claim_object, queries_evidence):
    claim = claim_object.get("claim", "")
    claim_date = claim_object.get("claim_date", "Unknown") or "Unknown"
    speaker = claim_object.get("speaker", "Unknown") or "Unknown"
    reporting_source = claim_object.get("reporting_source", "Unknown") or "Unknown"
    location_iso_code = claim_object.get("location_ISO_code", "Unknown") or "Unknown"
    evidence_block, prediction_evidence = build_examples_content(queries_evidence)

    prompt = (
        f'Claim: {claim}\n'
        f'Claim Date: {claim_date}\n'
        f'Claim Speaker: {speaker}\n'
        f'Location ISO Code: {location_iso_code}\n'
        f'Reporting Source: {reporting_source}\n'
        f'Queries and Evidence: \n{evidence_block}'
    )
    return prompt, prediction_evidence

def build_vp_prompt(claim_object, queries_evidence):
    task, q_evidence_list = create_user_task(claim_object, queries_evidence)

    messages = [
        {
          "role": "system",
          "content": "You are a highly capable, thoughtful, and precise fact-checker." + "\n" + instruction_examples
        },
        {
          "role": "user",
          "content": "Fact-check this claim:\n" + task + '''\nYou must carefully reason step-by-step using the context and evidence to determine the final label of the claim.
OUTPUT FORMAT:
Reasoning:
1. <concise rewrite of claim and intent; Note the timeline, location, statistics, numbers, quotes, events>
2. <evidence assessment>
3. <contradictions / gaps / biases noted>
4. <why the balance of evidence leads to the chosen label>

Label:
<Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking>'''
        }
    ]
    return messages, q_evidence_list

# Extract label from model output
def get_label_from_output(output: str) -> Optional[str]:
    if any(x in output for x in ["Not Enough Evidence", "Insufficient Evidence", "Missing Evidence", "NEI"]):
        return "Not Enough Evidence"
    elif any(x in output for x in ["Conflicting Evidence/Cherrypicking", "Cherrypicking", "Conflicting Evidence"]):
        return "Conflicting Evidence/Cherrypicking"
    elif any(x in output for x in ["Supported", "supported", "True"]):
        return "Supported"
    elif any(x in output for x in ["Refuted", "refuted", "False"]):
        return "Refuted"
    return None

if __name__ == "__main__":
    # python veracity_prediction.py --CLAIMS_PATH ./data_store/averitec/dev.json --OUTPUT_FILE ./output/dev_veracity_prediction.json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--CLAIMS_PATH', default='./data_store/averitec/dev.json')
    parser.add_argument("-o", "--OUTPUT_FILE", default="./output/dev_veracity_prediction.json")
    args = parser.parse_args()

    with open(args.CLAIMS_PATH, "r", encoding="utf-8") as fp:
        claims_dataset = json.load(fp)
        claims_count = len(claims_dataset)

    # Load Few-shot examples
    with open(EXAMPLES_VERACITY_PATH, "r", encoding="utf-8") as f:
        instruction_examples = f.read()

    root_dir = 'output'
    semantic_results_dir = f'{root_dir}/semantic_results'

    # Load Veracity LLM
    llm, tokenizer, sampling_params = llm_utils.init_vllm_llm_vp()

    predictions = []
    for batch_start in tqdm(range(0, claims_count, BATCH_SIZE_VP)):
        batch_end = min(batch_start + BATCH_SIZE_VP, claims_count)
        current_batch_ids = [k for k in range(batch_start, batch_end)]

        # Prepare batch inputs
        batch_texts = []
        batch_qvs = []
        batch_ids = []
        for index_id in current_batch_ids:
            file_path = os.path.join(semantic_results_dir, f'{index_id}.json')
            if os.path.exists(file_path):
                with open(file_path, encoding='utf-8') as fp:
                    queries_evidence = json.load(fp)

                messages, q_v = build_vp_prompt(claims_dataset[index_id], queries_evidence)
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_texts.append(text)
                batch_qvs.append(q_v)
                batch_ids.append(index_id)
            else:
                print(f"File not found: {file_path}")

        # Process batch
        outputs = llm.generate(batch_texts, sampling_params)

        # Process each result
        for index_id, text, q_v, output in zip(batch_ids, batch_texts, batch_qvs, outputs):
            claim_object = claims_dataset[index_id]
            gen_text = output.outputs[0].text.strip()
            label = get_label_from_output(gen_text.split('Label:')[-1])

            # Retry logic for failed parsing
            retry_count = 0
            max_retries = 3

            while label is None and retry_count < max_retries:
                retry_count += 1
                print(f"Index {index_id} - Retry {retry_count}/{max_retries}: Could not parse response.")

                # Individual retry for this specific claim
                retry_output = llm.generate([text], sampling_params)
                gen_text = retry_output[0].outputs[0].text.strip()
                label = get_label_from_output(gen_text.split('Label:')[-1])

            # Store result
            json_data = {
                "claim_id": claim_object.get('claim_id', index_id),
                "claim": claim_object["claim"],
                "evidence": q_v,
                "pred_label": label or "Refuted",
                "gold_label": claim_object.get("label", None),
                "llm_output": gen_text,
            }
            predictions.append(json_data)

        with open(args.OUTPUT_FILE, 'w', encoding="utf-8") as fp:
            json.dump(predictions, fp, indent=2)