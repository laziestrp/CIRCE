# SPDX-License-Identifier: Apache-2.0
"""Dataset prompts used by the main experiments and explicit prompt overrides."""
from .data import Example

PROMPT_VERSION = "main_prompts_v1"
DATASET_PROMPTS = {
    "gsm8k": 'Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".\n\nQuestion:\n{input_data}\n',
    "math": "Question: {input_data}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n",
    "mmlu": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nQuestion:\n{input_data}\n",
    "belebele": "Answer the following multiple choice reading-comprehension question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Please fully understand the passage and give explanations step by step before answering.\n\n{input_data}\n",
    "commonsenseqa": "Answer the following multiple choice common-sense reasoning question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE. Think step by step and output the reasoning process before answering.\n\n{input_data}",
    "theoremqa": '''
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Please read a math problem, and then think step by step to derive the answer. The answer is decided by Answer Type.
If the Answer type in [bool], the answer needs to be True or False.
Else if the Answer type in [integer, float] , The answer needs to be in numerical form.
Else if the Answer type in [list of integer, list of float] , the answer needs to be a list of number like [2, 3, 4].
Else if the Answer type in [option], the answer needs to be an option like (a), (b), (c), (d).
You need to output the answer in your final sentence like 'Therefore, the answer is ...'.

### Question:\x20
{input_data}

### Answer_type: {answer_type}

### Response:''',
}
EXTENSION_PROMPT = "Solve the following task carefully. Explain your reasoning, then give your final answer on a separate last line exactly as Answer: <option letter>.\n\n{task}"
for _dataset in ("logiqa2", "mmlu_pro", "race_high"):
    DATASET_PROMPTS[_dataset] = EXTENSION_PROMPT


def task_text(example: Example):
    text = example.question
    if example.passage:
        text = "Passage:\n" + example.passage + "\n\nQuestion: " + text
    elif example.options and example.dataset == "mmlu_pro":
        text = "Question: " + text
    if example.options:
        text += "\n" + "\n".join(f"{label}. {option}" for label, option in
                                 zip(example.choice_labels, example.options))
    return text


def build_prompt(example: Example):
    """Gold labels never enter prompts; an explicit prompt is used verbatim."""
    if example.prompt is not None:
        return example.prompt
    template = DATASET_PROMPTS[example.dataset]
    # Use literal replacement to preserve the historical MATH boxed{{}} text.
    return (template.replace("{input_data}", task_text(example))
            .replace("{task}", task_text(example))
            .replace("{answer_type}", example.answer_type))


def tokenize_prompt(tokenizer, prompt):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("a checkpoint chat template is required")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt",
    )
