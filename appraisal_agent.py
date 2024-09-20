# -*- coding: utf-8 -*-

import json
import numpy as np
import string
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers import AutoModelForCausalLM
if torch.cuda.is_available():
     model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
     tokenizer = AutoTokenizer.from_pretrained(model_id)
     model = AutoModelForCausalLM.from_pretrained(
         model_id,
         torch_dtype=torch.bfloat16,
         device_map="auto",
         )
     model.cuda()

lora_config = LoraConfig(
                r=32, #
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,  
                lora_dropout= 0.05 #0.1 #0.05
            )

PREDICT_INSTRUCTION = """
### Instruction:
Given a list of facts, predict the emotion behind the last utterance of {ticker}. Give your response in this format:
(1) Emotion Label: Choose from the following set: ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
(2) Deduction: Explain your reasoning by clearly stating the key facts and how they logically lead to the predicted emotion.

### Examples:
Here are some examples:
{examples}
(END OF EXAMPLES)

### Facts:
{summary}

### Emotion Label:"""



class PredictAgent:
    def __init__(self, ticker: str, summary: str, target: str, predict_llm=model, tokenizer=tokenizer) -> None:
        self.ticker = ticker
        self.summary = summary
        self.target = target
        self.prediction = ''

        self.predict_prompt = PREDICT_INSTRUCTION
        self.predict_examples = PREDICT_EXAMPLES

        self.llm = predict_llm
        self.tokenizer = tokenizer

        self.__reset_agent()


    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        facts = "Facts:\n" + self.summary + "\n\nEmotion Label: "
        self.scratchpad += facts
        # print(facts, end="")

        try:
            response = self.prompt_agent()
            self.scratchpad += response
            parsed_response = self.scratchpad.split('Emotion Label:')[-1].strip()
            # self.prediction = parsed_response
            self.prediction = parsed_response.split()[0]
            print("scratchpad----------------")
            print(self.prediction, end="\n\n\n\n")
            self.finished = True

        except Exception as e:
            print(f"Error during model prediction: {e}")

        # print("show-------------", self.target, self.prediction)
        self.finished = True



    def run_reflect(self, response, reset=True) -> None:
        # if reset:
        #     self.__reset_agent()
        try:
            # response = self.prompt_agent()
            self.scratchpad += response
            parsed_response = self.scratchpad.split('Emotion Label:')[-1].strip()
            # self.prediction = parsed_response
            self.prediction = parsed_response.split()[0]
            print("scratchpad-----aftter reflect-----------")
            print(self.prediction, end="\n\n\n\n")

            ass = self.is_correct()
            print("updated------", ass)

            self.finished = True

        except Exception as e:
            print(f"Error during model prediction: {e}")

        # print("show-------------", self.target, self.prediction)
        self.finished = True



    def prompt_agent(self) -> str:
        prompt = self._build_agent_prompt()
        print(f"Prompt----: {prompt}")

         # Tokenize the prompt and generate attention mask
        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

        outputs = self.llm.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=encoding.input_ids.size(1) + 150,  # Extend max_length by 150 tokens for the generated summary
                num_return_sequences=1
            )

        # Decode the outputs to get the string
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"Model response: {response}")

        return response

    def _build_agent_prompt(self) -> str:
        return self.predict_prompt.format(
            ticker=self.ticker,
            examples=self.predict_examples,
            summary=self.summary
        )

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        print("after reflection they are------:", self.prediction, self.target)

        return self.prediction.lower() == self.target.lower()

        # return EM(self.target, self.prediction)

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''


class NShotLLM:
    def __init__(self, model=None, tokenizer=None, reward_model=None, num_shots=4):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.num_shots = num_shots

    def queries_to_scores(self, list_of_strings):
        return [output["score"] for output in self.reward_model(list_of_strings)]

    def __call__(self, prompt):
        query = self.tokenizer.encode(prompt, return_tensors="pt")
        queries = query.repeat((self.num_shots, 1))
        output_ids = self.model.generate(
            queries,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=64,
        )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        scores = torch.tensor(self.queries_to_scores(output))
        output_ids = output_ids[scores.topk(1).indices[0]][len(query[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response



"""## this part is for showing the test case ##"""

REFLECTION_HEADER = 'You have attempted to tackle the following task before and failed.'


# Path: your_module.py
from typing import List

class PredictReflectAgent2(PredictAgent):
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 predict_llm,
                 reflect_llm,
                 tokenizer,
                 tokenizer2
                 ) -> None:

        super().__init__(ticker, summary, target, predict_llm, tokenizer)
        self.predict_llm = predict_llm
        self.reflect_llm = reflect_llm
        # self.agent_prompt = PREDICT_REFLECT_INSTRUCTION

        self.reflections = []

        self.tokenizer = tokenizer2
        self.reflections_str: str = ''

        self.reflection_generator = ReflectionGenerator(reflect_llm, tokenizer)
        self.emotion_label_generator = EmotionLabelGenerator(predict_llm, tokenizer2)

        self.target = target

        # self.prediction= '' #self.scratchpad.split('Emotion Label:')[-1]


    def run(self, reset=True) -> None:

        PredictAgent.run(self, reset=reset)

        # if self.is_finished() and not self.is_correct():
        #     model_output = self.reflect()

        #     # pred = model_output.split('Emotion Label:')[-1].split()[0]


        #     # print("correcting the mind after refleciotn-----------------", pred)


        #     PredictAgent.run_reflect(self, model_output, reset=True)


    # def run(self, reset=True) -> None:
        # if self.is_finished() and not self.is_correct():
        #     self.reflect()

            # pred = model_output.split('Emotion Label:')[-1].split()[0]


            # print("correcting the mind after refleciotn-----------------", pred)


            # PredictAgent.run_reflect(self, model_output, reset=True)


        # PredictAgent.run(self, reset=False)


    def reflect(self) -> str:
        print('Reflecting...\n')
        reflection = self.prompt_reflection()

        print("Reflection generated:", reflection)

        self.reflections += [reflection]
        self.reflections_str = format_reflections(self.reflections)

        print("Formatted reflections:")
        print(self.reflections_str, end="\n\n\n\n")

        model_output = self.emotion_label_generator.generate_emotion_label(context = self.ticker, facts=self.summary, reflections = self.reflections_str, examples=self.predict_examples )

        return model_output

    def prompt_reflection(self) -> str:

        reflection = self.reflection_generator.generate_reflection(
            previous_label=self.scratchpad.split('Emotion Label:')[-1].split()[0],
            context=self.ticker,
            previous_trial=self.scratchpad
        )
        return reflection

    def run_n_shots(self, model, tokenizer, reward_model, num_shots=4, reset=True) -> None:
        self.llm = NShotLLM(model, tokenizer, reward_model, num_shots)
        super().run(reset=reset)



class ReflectionGenerator:
    def __init__(self, reflect_llm, tokenizer):
        self.reflect_llm = reflect_llm
        self.tokenizer = tokenizer

    def generate_reflection(self, previous_label: str, context: str, previous_trial: str) -> str:
        prompt = (
            f"You will be given a previous trial in which you were given access to a list of facts to predict the emotion label of the last utterance of {context}."
            f"You were unsuccessful in deductive reasoning because you gave the wrong emotion label of {previous_label}."
            f"Re-evaluate the emotional context and overall tone of the last utterance. Use complete sentences.\n\n"
            f"Reflection:\n"
        )

        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

        outputs = self.reflect_llm.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=encoding.input_ids.size(1) + 100,
            num_return_sequences=1,
            use_cache=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
###this is for counterfactural
Conterfactural_reasoning_Prompt = """
Instruction:
What if the speaker's emotional response wasn't {previous_label}, but instead displayed a different emotion?

Steps:
Premises: Carefully re-examine each {premises}
Utterance: Identify key emotional indicators in {utterance}
Counterfactual Emotion:
a. Predict an emotion that contradicts the apparent emotion in the utterance
b. Analyze how this contradictory emotion could fit the situation
c. Explore the implications if the speaker were feeling this contradictory emotion

Response Format:
Emotion Label: [ONE label from: happy, sad, neutral, angry, excited, frustrated]
Explanation: [Your reasoning in 2-3 short sentences, including how the contradictory emotion fits the situation and its potential implications]
Your Response:
"""

# Path: your_module.py
from typing import List

class PredictReflectAgent2(PredictAgent):
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 dialog: str,
                 predict_llm,
                 reflect_llm,
                 tokenizer,
                 tokenizer2
                 ) -> None:

        #this part is for initializing the predictagent
        super().__init__(ticker, summary, target, dialog, predict_llm, tokenizer)
        self.predict_llm = predict_llm
        self.reflect_llm = reflect_llm
        # self.agent_prompt = PREDICT_REFLECT_INSTRUCTION

        self.reflections = []

        self.tokenizer = tokenizer2
        self.reflections_str: str = ''

        # self.reflection_generator = ReflectionGenerator(reflect_llm, tokenizer)
        self.emotion_label_generator = EmotionLabelGenerator(predict_llm, tokenizer)

        self.target = target

        self.dialog = dialog

        self.update_explanation = ''

        self.predict_examples = PREDICT_EXAMPLES

        # self.prediction = self.scratchpad.split('Emotion Label:')[-1].strip()

##############################this is the original one ##########################
    # def run(self, reset=True) -> None:
    #     if self.is_finished() and not self.is_corrects():
    #         self.reflect()
    #     else:
    #       PredictAgent.run(self, reset=reset)
#######################################################################################

    def run(self, reset=True) -> None:
        PredictAgent.run(self, reset=reset)


    def reflect_1(self, summary, label) -> None:
        print('-------------#################>>>>>>>>>>>Reflecting#################################...\n')
        print(label)
        print("#################################################")
        # model_output = self.emotion_label_generator.generate_emotion_label(context = self.ticker, facts = summary, previous_label=self.scratchpad.split('Emotion Label:')[-1].strip())

        model_output = self.emotion_label_generator.generate_emotion_label(context = self.ticker, facts = summary, previous_label=str(label))


        # self.update_explanation = model_output.split('Appraisal:')[-1].strip()
        self.update_explanation = model_output.split('Explanation:')[-1].strip()

        # response= model_output.split('Emotion:')[-1]
        response= model_output.split('Emotion Label:')[-1]

        self.prediction = response.split()[0].strip()

        print("correcting the mind after refleciotn---------------->>>>>>>-", self.prediction)
        print("do reasoning----------------", self.update_explanation)
        print("-------------------------------------------------------------------------------------------------")
        print("true is ----------------", self.target)

        outts = self.is_corrects()
        print("after reflection they are:", outts)



    def reflect_2(self) -> None:
        prompt_re = Re_PREDICT_INSTRUCTIONS.format(facts = self.summary, utterance = self.ticker)
         # Tokenize the prompt and generate attention mask
        encoding = self.tokenizer(prompt_re, return_tensors="pt", padding=True, truncation=True, max_length=2048)

        outputs = self.llm.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=encoding.input_ids.size(1) + 300,  # Extend max_length by 150 tokens for the generated summary
                num_return_sequences=1,
                temperature=0.7,
                use_cache=True
            )

        # response = self.tokenizer.batch_decode(outputs,skip_special_tokens=True)

        # Decode the outputs to get the string
        respond = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.prediction = respond.split()[0].strip()

        predict = respond.split('Emotion Label:')[-1]
        self.prediction = predict.split()[0].strip()

        self.update_explanation = respond.split('Explanation:')[-1].strip()

        print("------------correcting the mind after refleciotn-----------------", self.prediction)
        print("do reasoning----------------", self.update_explanation)
        print("true is ----------------", self.target)


    def run_n_shots(self, model, tokenizer, reward_model, num_shots=4, reset=True) -> None:
        self.llm = NShotLLM(model, tokenizer, reward_model, num_shots)
        super().run(reset=reset)

    def is_corrects(self) -> bool:
        return EMs(self.target, self.prediction)


def EMs(prediction, sentiment) -> bool:
    return prediction.strip().lower() == sentiment.strip().lower()


class EmotionLabelGenerator:
    def __init__(self, predict_llm, tokenizer):
        self.predict_llm = predict_llm
        self.tokenizer = tokenizer

    def generate_emotion_label(self, previous_label: str, context: str, facts: str,) -> str:

        prompt = Reflect_predict_INSTRUCTION.format(
            previous_label = previous_label,
            premises = facts,
            utterance=context)

        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

        outputs = self.predict_llm.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=encoding.input_ids.size(1) + 300,
            num_return_sequences=1,
            temperature=0.3,
            use_cache=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response


def format_reflections(reflections: List[str]) -> str:
    if not reflections:
        return ''
    else:
        # return header + '\nExplanation:\n- ' + '\n- '.join([r.strip() for r in reflections])
        # return '\nExplanation:\n- ' + '\n- '.join([r.strip() for r in reflections])
        # return '\nExplanation:\n- ' + '\n- '.join([r.strip() for r in reflections])
        return [r.strip() for r in reflections]



agent_cls = PredictReflectAgent2
agents = [agent_cls(row['ticker'], row['summary'], row['target'],row['dialog'], model, model, tokenizer, tokenizer) for _, row in data[63:].iterrows()]
print("Loaded Train Agents.")

import json
from tqdm import tqdm

num_reflect_trials = 1
datasets_dir = "./datasets/"

# Collect comparison data
comparison_data = []

for trial in tqdm(range(num_reflect_trials), desc="Trials"):
    print(f"\n{'#' * 40}")
    print(f"Let's get started with trial: {trial}")
    print(f"{'#' * 40}\n")

    for idx, agent in enumerate(tqdm(agents, desc="Agents")):
        rewards = 0
        current_state = ''
        update_state = ''
        done = False

        agent.run()
        current_state = agent.explanation
        previous_states = [current_state]  # Initialize as a list
        prev_lab = [agent.prediction]
        label = agent.prediction

        if agent.is_corrects():
            rewards = 0
            done = False
            update_state=''
        else:
            rewards = -1
            done = True
            agent.reflect_1(previous_states, prev_lab)
            label = agent.prediction

            update_state = agent.update_explanation
            previous_states.append(update_state)

            if agent.is_corrects():
                rewards += 1
                done = False
            else:
                prev_lab.append(label)
                rewards += -1
                done = True

        sample = {
            "user_input": agent.ticker,
            "state": current_state,
            "update_state": update_state,
            "actor_rewards": rewards,
            "action": agent.prediction,
            "target": agent.target,
            "done": done,
            "dialog": agent.dialog
        }

        comparison_data.append(sample)

        if done:
            agent.reflect_1(previous_states, prev_lab)
            update_state = agent.update_explanation
            previous_states.append(update_state)

            if agent.is_corrects():
                rewards += 1
                done = False
            else:
                rewards += -1
                done = True

            sample = {
                "user_input": agent.ticker,
                "state": current_state,
                "update_state": update_state,
                "actor_rewards": rewards,
                "action": agent.prediction,
                "target": agent.target.lower(),
                "done": done,
                "dialog": agent.dialog
            }

            comparison_data.append(sample)









"""
