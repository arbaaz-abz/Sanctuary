from helper import llm_utils
from helper.config import BATCH_SIZE_QG
from tqdm import *
import time
import random
import json
import os
import re
import ast
import argparse

def create_user_task(claim_object):
    claim = claim_object.get("claim", "")
    claim_date = claim_object.get("claim_date", "Unknown") or "Unknown"
    speaker = claim_object.get("speaker", "Unknown") or "Unknown"
    reporting_source = claim_object.get("reporting_source", "Unknown") or "Unknown"
    location_iso_code = claim_object.get("location_ISO_code", "Unknown") or "Unknown"
    
    prompt = (
        f'"claim": "{claim}",\n'
        f'"claim_date": "{claim_date}",\n'
        f'"claim_speaker": "{speaker}",\n'
        f'"location_ISO_code": "{location_iso_code}",\n'
        f'"reporting_source": "{reporting_source}"\n'
    )
    return prompt

def build_qg_prompt_v1(claim_object):
    messages = [
        {
          "role": "system",
          "content": "You are a highly capable, thoughtful, and precise fact-checker."
        },
        {
          "role": "user",
          "content": """Given a claim and its metadata, first classify the claim to the best of your knowledge and then generate relevant Question and Answer pairs. You must assign one of the following labels:
* Supported: The claim is fully backed by clear, consistent evidence.
* Refuted: The claim is directly contradicted by reliable evidence, or there is no concrete evidence to support the claim. 
* Conflicting Evidence/Cherrypicking: The evidence has both supporting and opposing arguements for the claim or is selectively presented to favour the claim.

Guidelines:
- Generate anywhere between 5 to 10 Q-A pairs. Generate more questions complex and multi-fact claims.
- Ensure questions have context. E.g. Given the claim "Every week, at least 12 doctors leave Nigeria to seek employment in the UK.", the question "When was the claim made regarding Nigerian doctors moving to the UK to seek employment?" has more context than the overly generic - "When was the claim made?".
- A question can have multiple sub-answers derived from various sources.
- Keep your answers brief (not more than 3 sentences).
- Fabricate random facts, figures, statements or arguements, if and when needed. 

OUTPUT FORMAT:
"label": <Supported, Refuted, Conflicting Evidence/Cherrypicking>,
"evidence": ```json{Question: [Answer-1, Answer-2, ...], Question: [Answer-1], ...}```

Example Input:
"claim": "82% of the public are wearing face masks on Translink buses and trains.",
"claim_date": "6-8-2020",
"claim_speaker": "@deptinfra",
"location_ISO_code": "GB",
"reporting_source": "twitter"

Example Output:
"label": "Supported",
"evidence": 
```json
{
    "Are face masks mandatory on buses and trains in the UK ?": [
        "Masks will be made mandatory for anyone in England wanting to use public transport, including the Tube, buses and trains from 15 June onwards.\n\nTravellers will be refused access to public transport if they do not comply with these rules, which have been put in place to reduce the spread of coronavirus, as more people head back to work.\n\nThe government has said face coverings should be worn on trains, underground trains, buses, ferries and planes. But people won\u2019t have to wear them inside railway stations or at bus terminals. UK transport secretary Grant Shapps said tonight: \u201cI can announce that as of Monday 15 June, face coverings will become mandatory on public transport.\n\n\u201cThe evidence suggests that wearing a face mask offers some, limited protection. It\u2019s a condition of travel. You cannot travel if you are not wearing a face covering.\u201d\n\nHowever, there will be exemptions for anyone with breathing difficulties and young children."
    ],
    "Are there people exempted from the use of face masks face coverings on buses in the United Kingdom?": [
        "You do not need to wear a face covering if you have a legitimate reason not to or are exempt on other grounds - these include the following:\n\n    not being able to put on, wear or remove a face covering because of a physical or mental illness or impairment, or disability\n    if you have a medical condition that prevents you from wearing a mask such as respiratory illness\n    if putting on, wearing or removing a face covering will cause you severe distress\n    if you are travelling with or providing assistance to someone who relies on lip reading to communicate\n    to avoid harm or injury, or the risk of harm or injury, to yourself or others\n    to avoid injury, or to escape a risk of harm, and you do not have a face covering with you\n    to eat or drink, but only if you need to and replace your face covering as soon as possible\n    to take medication\n    if a police officer or other official requests you remove your face covering\n    a child who is under the age of 11\n    any employee of the operator, or anyone contracted to provide a service on behalf of the operator who is performing their duty - although as mentioned above, reasonable alternative measures should be in place unless not possible to do so.\n    a member of the emergency services who are performing their duties\n\nPlease remember that there may be people on the list above who are exempt from wearing a face covering. Not all reasons are visible so please be mindful of others who may not be wearing a mask for a legitimate reason. \n\nOperators such as ourselves have been given discretion on how they enforce the new requirement to wear a face covering with the preferred method being \u201cengage, explain, encourage\u201d. Our drivers are not police officers and so are not there to enforce the law, but by following the Government guidelines, we are trying to engage people to explain the new law and then encourage them to wear face coverings in future."
    ],
    "Which area is specifically covered by the claim mentioning face mask usage on Translink public transport?": [
        "Translink buses and trains in Northern Ireland."
    ],
    "Are there people exempted from the use of face masks face coverings on public transport?": [
        "You do not need to wear a face covering if you have a legitimate reason not to or are exempt on other grounds"
    ],
    "Are face masks mandatory on buses and trains in Northern Ireland?": [
        "face coverings became mandatory on public transport from Friday, 10 July, in Northern Ireland"
    ],
    "What percent of the public were wearing face masks on Translink buses and trains by August 2020 when the claim was made?": [
        "compliance rates were around 80%"
    ]
}
```
""" + "\n" + "Now process this claim:\n" + create_user_task(claim_object)
        }
    ]
    return messages

# def build_qg_prompt_v2(claim_object):
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a highly capable, creative, and precise fact-checker. "
#                 "Your task is to first analyze a claim and classify it based on your general knowledge, "
#                 "and then generate investigative questions and plausible, simulated answers designed to "
#                 "help guide a real fact-checking process."
#             )
#         },
#         {
#             "role": "user",
#             "content": f"""Given the following claim and its metadata:
# {create_user_task(claim_object)}
# First, classify the claim to the best of your ability using your general knowledge, assigning one of the following labels:
# *   **Supported**: The claim seems generally plausible or aligns with common knowledge/likely scenarios.
# *   **Refuted**: The claim seems unlikely, contradicts common knowledge, or contains elements that are likely false.
# *   **Conflicting Evidence/Cherrypicking**: The claim is nuanced, involves potentially selective information, or addresses a topic where conflicting information is common.

# Then, generate a set of relevant Question and Answer (Q&A) pairs designed to thoroughly investigate this claim. These Q&A pairs will later help guide the search for real evidence.

# **Guidelines for Q&A Generation:**

# 1.  **Quantity:** Generate 5-10 Q&A pairs. Generate more questions for complex, nuanced, or multi-part claims.
# 2.  **Question Focus:**
#     *   Probe the **core assertion(s)** of the claim.
#     *   Target **specific details**: numbers, quantities, dates, timelines, locations, comparisons, involved parties, cause-effect relationships.
#     *   Consider questions about **definitions** of key terms used in the claim.
#     *   Formulate questions about the **scope** (e.g., geographical area, time period) the claim applies to.
#     *   Include questions that might explore potential **counter-arguments or alternative perspectives**.
# 3.  **Question Clarity & Context:**
#     *   Each question **must** include enough context from the claim to be unambiguous and specific. Avoid overly generic questions.
#     *   *Good Example:* For the claim "Every week, at least 12 doctors leave Nigeria to seek employment in the UK.", the question "When was the data collected regarding the 12 doctors leaving Nigeria weekly for the UK?" is better than "When was the data collected?".
# 4.  **Answer Simulation:**
#     *   Keep answers concise (typically 1-3 sentences).
#     *   The answers **must be fabricated** but should **simulate plausible evidence snippets, facts, figures, or statements** relevant to the question.
#     *   Crucially, these simulated answers should contain **keywords, entities, concepts, and plausible details** that would likely appear in real-world documents addressing the question, making them useful for downstream retrieval tasks.
#     *   Do *not* simply restate the question or be evasive. Provide a concrete (though fabricated) piece of information.
# 5.  **Overall Output Format:**
#     *   Provide the classification label first, followed by the JSON object containing the Q&A pairs, enclosed in triple backticks.
#     *   The format must be exactly:
#         "label": <Supported, Refuted, Conflicting Evidence/Cherrypicking>,
#         "evidence": ```json{{ ... Q&A dictionary ... }}```
#     *   Ensure the evidence part is a valid JSON object string: `{{"Question 1": ["Simulated Answer 1"], "Question 2": ["Simulated Answer 2a", "Simulated Answer 2b"], ...}}`

# **Example Input:**
# "claim": "82% of the public are wearing face masks on Translink buses and trains.",
# "claim_date": "6-8-2020",
# "claim_speaker": "@deptinfra",
# "location_ISO_code": "GB",
# "reporting_source": "twitter"

# **Example Output:**
# "label": "Supported",
# "evidence":
# ```json
# {{
#     "What specific geographical area does the claim about 82% face mask usage on Translink pertain to?": [
#         "The claim specifically refers to Translink public transport services operating within Northern Ireland."
#     ],
#     "According to the source, what methodology was used to determine the 82% face mask compliance rate on Translink services around August 2020?": [
#         "The 82% figure was based on observational surveys conducted by Translink staff on a sample of bus and train routes across Northern Ireland during the first week of August 2020."
#     ],
#     "Was wearing face masks mandatory on public transport in Northern Ireland on August 6, 2020?": [
#         "Yes, face coverings became mandatory on public transport in Northern Ireland starting from July 10, 2020, requiring passengers to wear them unless exempt."
#     ],
#     "Are there official figures or surveys corroborating the 82% mask usage claim on Translink around August 2020?": [
#         "Department for Infrastructure (DfI) statements around that time indicated high compliance levels, generally reported as being 'around 80%' on public transport following the mandate."
#     ],
#     "Did the 82% compliance rate for face masks on Translink represent an increase or decrease compared to previous weeks in July 2020?": [
#         "Reports suggested this represented a steady rate, consistent with levels observed shortly after the mandatory requirement was introduced in mid-July 2020."
#     ],
#     "Were there any specific exemptions to the mandatory face mask rule on Translink services in August 2020?": [
#         "Exemptions were in place for individuals with certain health conditions, disabilities, children under 11, and transport staff in specific circumstances, as per Northern Ireland government guidelines."
#     ]
# }}
#     """
#         }
#     ]
#     return messages

def parse_response(gen_text):
    try:
        evidence = ast.literal_eval(gen_text.split('```json')[1].split('```')[0])
        return evidence
    except Exception as e:
        return {}

if __name__ == "__main__":
    # CLAIMS_PATH - /home/rogers/thesis/AVeriTeC/data_store/averitec/dev.json
    # python question_doc_generator.py --CLAIMS_PATH ./data_store/averitec/dev.json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--CLAIMS_PATH', default='./data_store/averitec/dev.json')
    args = parser.parse_args()

    with open(args.CLAIMS_PATH, "r", encoding="utf-8") as fp:
        claims_dataset = json.load(fp)
        claims_count = len(claims_dataset)
    
    llm, tokenizer, sampling_params = llm_utils.init_vllm_llm_qg()
    
    root_dir = "output"
    os.makedirs(root_dir, exist_ok=True)
    save_file_path = f'{root_dir}/qg_data.json'
    
    qg_output_data = {}
    for batch_start in tqdm(range(0, claims_count, BATCH_SIZE_QG)):
        batch_end = min(batch_start + BATCH_SIZE_QG, claims_count)
        current_batch_ids = [k for k in range(batch_start, batch_end)]
            
        # Prepare batch inputs
        batch_texts = []
        for index_id in current_batch_ids:
            claim_object = claims_dataset[index_id]
            messages = build_qg_prompt_v1(claim_object)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_texts.append(text)
        
        # Process batch
        outputs = llm.generate(batch_texts, sampling_params)
        
        # Process each result
        for index_id, text, output in zip(current_batch_ids, batch_texts, outputs):
            gen_text = output.outputs[0].text.strip()
            evidence = parse_response(gen_text)
            
            # Retry logic for failed parsing
            retry_count = 0
            max_retries = 3
            
            while evidence == {} and retry_count < max_retries:
                retry_count += 1
                print(f"Index {index_id} - Retry {retry_count}/{max_retries}: Could not parse response.")
                
                # Individual retry for this specific claim
                retry_output = llm.generate([text], sampling_params)
                gen_text = retry_output[0].outputs[0].text.strip()
                evidence = parse_response(gen_text)
            
            # Store result
            qg_output_data[index_id] = evidence
            
            if evidence == {}:
                print(f"Index {index_id} - Failed after {max_retries} attempts.")

    # Write at the end of the batch
    with open(save_file_path, 'w', encoding="utf-8") as fp:
        json.dump(qg_output_data, fp, indent=4)