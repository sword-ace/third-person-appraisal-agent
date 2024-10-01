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
                r=16, #16
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=16, 
                lora_dropout= 0.05 #0.05 #0.1 #0.05
            )



AppraisalGenerator_PROMPT = """
Instruction: Deduce the emotion behind the given utterance, using ONLY the provided dialog and premises. Do not make assumptions beyond the given information. Respond using ONLY the following format:

Emotion Label: [choose one from: happy, sad, neutral, angry, excited, frustrated]
Appraisal: [Your reasoning in 2-3 short sentences]

Utterance: {utterance}
Dialog context: {dialog}
Premises: {knowledge}

Your Response:
"""



class AppraisalAgent:
    def __init__(self, utter: str, knowledge: str, target: str, predict_llm=model, tokenizer=tokenizer) -> None:
        self.utter = utter
        self.knowledge = knowledge
        self.target = target
        self.prediction = ''

        self.predict_prompt = AppraisalGenerator_PROMPT
        
        self.llm = predict_llm
        self.tokenizer = tokenizer

        self.__reset_agent()


    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        knowledge = "Knowledge:\n" + self.knowledge + "\n\nEmotion Label: "
        self.scratchpad += knowledge
        # print(knowledge, end="")

        try:
            response = self.prompt_agent()
            self.scratchpad += response
            parsed_response = self.scratchpad.split('Emotion Label:')[-1].strip()
          
            self.prediction = parsed_response.split()[0]
            print("scratchpad----------------")
            print(self.prediction, end="\n\n\n\n")
            self.finished = True

        except Exception as e:
            print(f"Error during model prediction: {e}")

        # print("show-------------", self.target, self.prediction)
        self.finished = True


    def run_reflect(self, response, reset=True) -> None:
       
        try:
            # response = self.prompt_agent()
            self.scratchpad += response
            parsed_response = self.scratchpad.split('Emotion Label:')[-1].strip()
            # self.prediction = parsed_response
            self.prediction = parsed_response.split()[0]
            print("scratchpad-----after reflect-----------")
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
                max_length=encoding.input_ids.size(1) + 150,  # Extend max_length by 150 tokens for the generated appraisal
                num_return_sequences=1
            )

        # Decode the outputs to get the string
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"Model response: {response}")

        return response

    def _build_agent_prompt(self) -> str:
        return self..format(
            utter=self.utter,
            examples=self.predict_examples,
            knowledge=self.knowledge
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


CounterfactualReasoning_PROMPT = """
Instruction:
What if the speaker's emotional response wasn't {previous_label}, but instead displayed a different emotion?

Steps:
1. Premises: Carefully re-examine each {knowledge}
2. Utterance: Identify key emotional indicators in {utterance}
3. Counterfactual Emotion:
    a. Predict an emotion that contradicts the apparent emotion in the utterance
    b. Analyze how this contradictory emotion could fit the situation
    c. Explore the implications if the speaker were feeling this contradictory emotion

Response Format:
Emotion Label: [choose one from: happy, sad, neutral, angry, excited, frustrated]
Appraisal: [Your reasoning in 2-3 short sentences]

Your Response:
"""


#####################initiate performing counterfactual reasoning####################

# Path: your_module.py
from typing import List

class ReflectAgent(AppraisalAgent):
    def __init__(self,
                 utter: str,
                 knowledge: str,
                 target: str,
                 dialog: str,
                 predict_llm,
                 reflect_llm,
                 tokenizer,
                 tokenizer2
                 ) -> None:

        #this part is for initializing the appraisal agent
        super().__init__(utter, knowledge, target, dialog, predict_llm, tokenizer)
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


    def reflect_1(self, knowledge, label) -> None:
        print('-------------#################>>>>>>>>>>>Reflecting#################################...\n')
        print(label)
        print("#################################################")
        
        model_output = self.emotion_label_generator.generate_emotion_label(context = self.utter, knowledge = knowledge, previous_label= label)

        self.update_explanation = model_output.split('Appraisal:')[-1].strip()

        # response= model_output.split('Emotion:')[-1]
        response= model_output.split('Emotion Label:')[-1]

        self.prediction = response.split()[0].strip()

        print("correcting the mind after reflection---------------->>>>>>>-", self.prediction)
        print("do reasoning----------------", self.update_explanation)
        print("-------------------------------------------------------------------------------------------------")
        print("true is ----------------", self.target)

        outts = self.is_corrects()
        print("after reflection they are:", outts)


    def is_corrects(self) -> bool:
        return EMs(self.target, self.prediction)


def EMs(prediction, sentiment) -> bool:
    return prediction.strip().lower() == sentiment.strip().lower()


class EmotionLabelGenerator:
    def __init__(self, predict_llm, tokenizer):
        self.predict_llm = predict_llm
        self.tokenizer = tokenizer

    def generate_emotion_label(self, previous_label: str, context: str, knowledge: str,) -> str:

        prompt = CounterfactualReasoning_PROMPT .format(
            previous_label = previous_label,
            premises = knowledge,
            utterance=context)

        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

        outputs = self.predict_llm.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=encoding.input_ids.size(1) + 250,
            num_return_sequences=1,
            temperature=0.8,
            use_cache=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response


def format_reflections(reflections: List[str]) -> str:
    if not reflections:
        return ''
    else:
        # return header + '\nAppraisal:\n- ' + '\n- '.join([r.strip() for r in reflections])
        # return '\nAppraisal:\n- ' + '\n- '.join([r.strip() for r in reflections])
        # return '\nAppraisal:\n- ' + '\n- '.join([r.strip() for r in reflections])
        return [r.strip() for r in reflections]


agent_cls =ReflectAgent
agents = [agent_cls(row['utter'], row['knowledge'], row['target'],row['dialog'], model, model, tokenizer, tokenizer) for _, row in data[:50].iterrows()]
print("Loaded Train Agents.")

######################store in appraisal trajectories##############################

import json
from tqdm import tqdm

num_reflect_trials = 1
datasets_dir = "./datasets/"

# Collect comparison data
comparison_data = []

collect = []
collect1 = []
collect2 = []
for trial in tqdm(range(num_reflect_trials), desc="Trials"):
    counter=0
    counter_1 = 0
    counter_2 = 0
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
            counter+=1
        else:
            rewards = -1
            done = True
            # agent.reflect_1(previous_states, prev_lab)

            agent.reflect_1(agent.update_explanation, prev_lab)

            label = agent.prediction

            update_state = agent.update_explanation
            previous_states.append(update_state)

            if agent.is_corrects():
                rewards += 1
                done = False
                counter_1 +=1

            else:
                prev_lab.append(label)
                rewards += -1
                done = True

        sample = {
            "user_input": agent.utter,
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
            # agent.reflect_1(previous_states, prev_lab)
            agent.reflect_1(agent.update_explanation, prev_lab)
            update_state = agent.update_explanation
            previous_states.append(update_state)

            if agent.is_corrects():
                rewards += 1
                done = False
                counter_2 += 1
            else:
                rewards += -1
                done = True

            sample = {
                "user_input": agent.utter,
                "state": current_state,
                "update_state": update_state,
                "actor_rewards": rewards,
                "action": agent.prediction,
                "target": agent.target.lower(),
                "done": done,
                "dialog": agent.dialog
            }

            comparison_data.append(sample)

        collect.append(counter)
        collect1.append(counter_1)
        collect2.append( counter_2)


###evaluatorLLM###########
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words("english")

import json
import numpy as np
import string
from tqdm import tqdm

class EmotionAnalyzer:
    def __init__(self, vad_filepath, json_file, stopwords, predict_llm, tokenizer):
        self.vad_dict = self.load_nrc_vad_lexicon(vad_filepath)
        self.json_data = self.load_json_data(json_file)
        self.stop_words = stopwords
        self.predict_llm = predict_llm
        self.tokenizer = tokenizer

    def load_nrc_vad_lexicon(self, filepath):
        print("Loading VAD lexicon...")
        with open(filepath, 'r') as file:
            vad_dict = json.load(file)
        print("VAD lexicon loaded successfully.")
        return vad_dict

    def load_json_data(self, filepath):
        print("Loading JSON data...")
        with open(filepath, 'r') as file:
            json_data = json.load(file)
        print("JSON data loaded successfully.")
        return json_data

    def tokenize(self, text):
        return text.lower().split()

    def clean_response(self, response):
        # Remove punctuation and strip whitespace for a robust comparison
        return response.strip().lower().translate(str.maketrans('', '', string.punctuation))

     # the min_max is computed based on all datasets
    def min_max_normalize(self, values, new_min=-1, new_max=1):

        print(values)
        old_min = min(values)
        old_max = max(values)

        if old_min == old_max:
            return [new_min + (new_max - new_min) / 2] * len(values)

        normalized_values = [
            new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)
            for value in values
        ]
        return normalized_values

    def compute_vad_scores(self, utterances):
        total_val = []
        total_arous = []
        print("Computing VAD scores...")
        for utterance in tqdm(utterances, desc="Processing utterances"):
            tokens = [token for token in self.tokenize(utterance) if token not in self.stop_words]
            valence_scores = []
            arousal_scores = []

            for token in tokens:
                if token in self.vad_dict:
                    valence_scores.append(self.vad_dict[token][0])
                    arousal_scores.append(self.vad_dict[token][1])
                # else:
                #     valence_scores.append(0)
                #     arousal_scores.append(0)

            avg_valence = round(np.mean(valence_scores), 2)
            avg_arousal = round(np.mean(arousal_scores), 2)

            total_val.append(avg_valence)
            total_arous.append(avg_arousal)

        return total_val, total_arous

    def generate_emotion_label(self, emo, valence, arousal):
        Evaluation_PROMPT = f'''
                Given the range of the {emo} class in the Circumplex Model of Affect, do the valence score of {valence} and the arousal score of {arousal} together fit within this range?

                Answer only 'yes' or 'no'.
                '''

        prompt = Evaluation_PROMPT.format(
            emo=emo,
            valence=valence,
            arousal=arousal)

        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

        outputs = self.predict_llm.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=encoding.input_ids.size(1) + 10,
            num_return_sequences=1)


        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        resd = response.split('Answer:')[-1].strip()

        # resd = response.split()
        print("------------this gonna show yes or no--------",valence, arousal, emo)

        try:
          print("------------this gonna show yes or no--------", resd.split()[0])
          cleaned_response = self.clean_response(resd.split()[0])
          reward = 0 if cleaned_response == "yes" else -1


        except IndexError:
          print("Error: resd is empty or contains only whitespace")
          cleaned_response = ""
          reward = -1  # or whatever default value makes sense in your context

        # print("------------this gonna show yes or no--------", resd.split()[0])


        # cleaned_response = self.clean_response(resd.split()[0])
        # reward = 0 if cleaned_response == "yes" else -1

        return reward

    def process_json_data(self):
        json_data = self.json_data
        new_json_data = []
        all_states = []
        all_update_states = []

        print("Collecting states and update states...")
        for item in tqdm(json_data, desc="Collecting states"):
            if item['done']:
                # print(item)

                all_states.append(item['state'])
                if item.get('update_state'):
                  all_update_states.append(item['update_state'])
            else:
                # print("--------------------", item['done'])

                if item.get('update_state'):
                  all_states.append(item['update_state'])
                else:
                  all_states.append(item['state'])

        # Compute VAD scores for all states and update_states
        state_valences, state_arousals = self.compute_vad_scores(all_states)
        update_valences, update_arousals = self.compute_vad_scores(all_update_states)

        norm_state_valence = self.min_max_normalize(state_valences)
        norm_state_arousal = self.min_max_normalize(state_arousals)
        norm_update_valence = self.min_max_normalize(update_valences)
        norm_update_arousal = self.min_max_normalize(update_arousals)

        update_index = 0 #####while update_valence and update_arousal are only computed for 'done' items.
        print("Processing items and generating emotion labels...")
        for i, item in tqdm(enumerate(json_data), desc="Processing items", total=len(json_data)):
            new_item = item.copy()
            emotion_label = item['target']

            if item['done']:
                state_valence = norm_state_valence[i]
                state_arousal = norm_state_arousal[i]


                # Check if we still have updates available
                if update_index < len(norm_update_valence):
                    update_valence = norm_update_valence[update_index]
                    update_arousal = norm_update_arousal[update_index]

                    # Generate emotion labels for both
                    state_reward = self.generate_emotion_label(emotion_label, state_valence, state_arousal)
                    update_reward = self.generate_emotion_label(emotion_label, update_valence, update_arousal)

                    # Use the worse reward
                    reward = state_reward + update_reward
                    update_index += 1

            else:
                state_valence = norm_state_valence[i]
                state_arousal = norm_state_arousal[i]
                reward = self.generate_emotion_label(emotion_label, state_valence, state_arousal)

            # Add the critic_rewards to the new item
            new_item['critic_rewards'] = reward
            new_json_data.append(new_item)

        return new_json_data


