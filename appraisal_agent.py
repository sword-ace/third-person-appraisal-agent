# -*- coding: utf-8 -*-

import json
import numpy as np
import string
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers import AutoModelForCausalLM
if torch.cuda.is_available():
     model_id = "mistralai/Mistral-7B-Instruct-v0.3"
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



AppraisalInstruction_Prompt = """
Analyze the given utterance within its dialogue context. Provide a concise appraisal and predict an emotion label in the following format:

1. Highlight the key elements of the situation from the dialogue.
2. Evaluate how the utterance aligns with the speaker’s goals and expectations.
3. Combine (1) and (2) to determine whether the utterance supports, contradicts or is neutral towards those goals and expectations.

Keep each section to 1-2 sentences. Base your analysis solely on the provided dialogue. 
Dialogue context: {dialog}
Utterance to analyze: {utterance}

Response Format:
Emotion Label: [Select one from: happy, sad, neutral, angry, excited, frustrated]
Explanation: [Step-by-step concise appraisal based on the points above]

Response:
"""


class PredictAgent:
    def __init__(self, ticker: str, target: str, dialog: str, predict_llm=model, tokenizer=tokenizer) -> None:
        self.ticker = ticker
        self.target = target
        self.dialog = dialog
        self.prediction = ''
        self.predict_prompt = AppraisalInstruction_Prompt
        self.llm = predict_llm
        self.tokenizer = tokenizer
        self.explanation = ''
        self.__reset_agent()

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()
        self.scratchpad = self.prompt_agent()

        response = self.scratchpad.split('Emotion Label:')[-1].strip()

        self.prediction = response.split()[0].strip()
        print("show-------------", self.prediction)

        self.finished = True

    def prompt_agent(self) -> str:
        prompt = self._build_agent_prompt()
        print(f"Prompt----: {prompt}")


        # Tokenize the prompt and generate attention mask
        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=encoding.input_ids.size(1) + 300,  # Extend max_length by 150 tokens for the generated summary
                num_return_sequences=1,
                temperature = 0.7,
                use_cache=True
            )


        # Decode the outputs to get the string
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"Model response: {response}")

        return response


    def _build_agent_prompt(self) -> str:
        return self.predict_prompt.format(
            utterance = self.ticker,
            dialog = self.dialog)


    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.target, self.prediction)

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''

def EM(prediction, sentiment) -> bool:
    return prediction.lower() == sentiment.lower()



tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


CounterfactualReasoning_Prompt= """
You made a wrong prediction, please perform a counterfactual analysis for the target utterance to refine your understanding of the speaker’s emotional state. Follow these steps to guide your thinking:
1. Reflect on why predictions in {previous_label} mismatch between the predictions and the speaker’s intentions or desires based on the target utterance.
2. Imagine an alternative emotion that better aligns with the speaker’s intentions and desires based on the dialogue.

Keep your analysis concise and structured. Use this counterfactual analysis to propose a more accurate emotion label that fits the given context.

Dialogue context: {dialog}
Utterance to analyze: {utterance}

Response Format:
Emotion Label: [choose one from: happy, sad, neutral, angry, excited, frustrated] 
Explanation: [Step-by-step concise appraisal based on the points above]


Response:
"""


from typing import List


#####################initiate performing counterfactual reasoning####################


class ReflectAgent(PredictAgent):
    def __init__(self,
                 ticker: str,
                 target: str,
                 dialog: str,
                 predict_llm,
                 reflect_llm,
                 tokenizer,
                 tokenizer2
                 ) -> None:

        #this part is for initializing the predictagent
        super().__init__(ticker, target, dialog, predict_llm, tokenizer)
        self.predict_llm = predict_llm
        self.reflect_llm = reflect_llm

        self.reflections = []

        self.tokenizer = tokenizer2
        self.reflections_str: str = ''

        self.emotion_label_generator = EmotionLabelGenerator(predict_llm, tokenizer)

        self.target = target

        self.dialog = dialog

        self.update_explanation = ''



    def run(self, reset=True) -> None:
        PredictAgent.run(self, reset=reset)


    def reflect_1(self,  label) -> None:
        print('-------------#################>>>>>>>>>>>Reflecting#################################...\n')
 
        model_output = self.emotion_label_generator.generate_emotion_label(utter = self.ticker, dialog= self.dialog, previous_label= label)

        self.update_explanation = model_output.split('Analysis:')[-1].strip()
    
        response= model_output.split('Emotion Label:')[-1]

        self.prediction = response.split()[0].strip()

        outts = self.is_corrects()
        print("after reflection they are:", outts)


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

    def generate_emotion_label(self, previous_label: str, utter: str, dialog: str,) -> str:


        prompt = CounterfactualReasoning_Prompt.format(
             previous_label = previous_label,
            dialog = dialog,
            utterance=utter)

        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        outputs = self.predict_llm.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=encoding.input_ids.size(1) + 300,
            num_return_sequences=1,
            temperature=0.7,
            use_cache=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response


def format_reflections(reflections: List[str]) -> str:
    if not reflections:
        return ''
    else:
        return [r.strip() for r in reflections]


agent_cls =ReflectAgent
agents = [agent_cls(row['ticker'], row['target'],row['dialog'], model, model, tokenizer, tokenizer) for _, row in data.iterrows()]
print("Loaded Train Agents.")

######################store in appraisal trajectories##############################

import json
from tqdm import tqdm

num_reflect_trials = 1
datasets_dir = "./datasets/"

# Collect data
comparison_data = [] 
predict_final = []

collect_reflections = {2: [], 3: [], 4: [], 5: []} 
correct_no_reflection = [] 

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
        prev = current_state

        # previous_states = [current_state]  
        prev_lab = [agent.prediction]
        label = agent.prediction

        if agent.is_corrects():
            rewards = 0
            done = False
            correct_no_reflection.append({
                "trial": trial,
                "agent_id": idx,
                "reflection_round": 0, 
                "user_input": agent.ticker,
                "state": current_state,
                "update_state": current_state,
                "actor_rewards": rewards,
                "action": agent.prediction,
                "target": agent.target,
                "done": done,
                "dialog": agent.dialog
            })
            predict_final.append(agent.prediction)
        else:
            rewards = -1
            done = True


            for reflection_round in range(1, 6):  
                agent.reflect_1(prev_lab)  
                label = agent.prediction
                # previous_states = agent.update_explanation
                prev = update_state
                update_state = agent.update_explanation

                if agent.is_corrects():
                    rewards = 0
                    done = False
                    predict_final.append(agent.prediction)


                    break
                else:
                    rewards -= 1  
                    # prev_lab = label
                    prev_lab.append(label)  # add wrong predictions to prev_lab

                    # current_state = agent.update_explanation


              
                if reflection_round >= 2:  # reflection starts from 2nd round
                    collect_reflections[reflection_round].append({
                        "trial": trial,
                        "agent_id": idx,
                        "reflection_round": reflection_round,
                        "user_input": agent.ticker,
                        "state": prev,
                        "update_state": update_state,
                        "actor_rewards": rewards,
                        "action": agent.prediction,
                        "target": agent.target,
                        "done": done,
                        "dialog": agent.dialog
                    })
                    
                if done is False:
                    break

for reflection_round in collect_reflections:
    collect_reflections[reflection_round].extend(correct_no_reflection)


########evaluatorLLM###########
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

if torch.cuda.is_available():
     model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
     tokenizer = AutoTokenizer.from_pretrained(model_id)
     model_eval = AutoModelForCausalLM.from_pretrained(
         model_id,
         torch_dtype=torch.bfloat16,
         device_map="auto",
         )
     model_eval.cuda()

lora_config = LoraConfig(
                r=16, #
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=16,  
                lora_dropout= 0.05 #0.1 #0.05



model_eval = get_peft_model(model_eval, lora_config)
print("Using LoRA")
model_eval.print_trainable_parameters()


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

#######################this part is for Reft framework ##################
class DummyDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.rewards = None
        self.critic_rewards = None
        self.states = None
        self.dialogs = None
        self.next_states = None
        self.dones = None
        self.batch_size = batch_size
        self.actions = None
        self.target = None

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observation": self.observations[rand_indices],
            "state": self.states[rand_indices],
            "dialog": self.dialogs[rand_indices],
            "action": self.actions[rand_indices],
            "reward": self.rewards[rand_indices],
            "critic_reward": self.critic_rewards[rand_indices],
            "next_state": self.next_states[rand_indices],
            "done": self.dones[rand_indices],
            "target": self.target[rand_indices],
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation,
        state,
        dialog,
        action,
        reward: np.ndarray,
        critic_reward: np.ndarray,
        next_state,
        done: np.ndarray,
        target,
        **kwargs
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(target, (float, int)):
            target = np.array(target)
        if isinstance(done, bool):
            done = np.array(done)
        # print(next_observation)
        # if isinstance(prompt_actionaction, int):
        #     action = np.array(action, dtype=np.int64)

        if self.observations is None:
            self.observations = np.array(['']*self.max_size, dtype = 'object')
            self.states =  np.array(['']*self.max_size, dtype = 'object')
            self.dialogs =  np.array(['']*self.max_size, dtype = 'object')
            self.actions = np.array(['']*self.max_size, dtype = 'object')
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.critic_rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_states = np.array(['']*self.max_size, dtype = 'object')
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)
            self.target = np.empty((self.max_size, *target.shape), dtype=target.dtype)

        assert reward.shape == ()
        assert done.shape == ()

        self.observations[self.size % self.max_size] = observation
        self.states[self.size % self.max_size] = state
        self.dialogs[self.size % self.max_size] = dialog
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.critic_rewards[self.size % self.max_size] = critic_reward
        self.next_states[self.size % self.max_size] = next_state
        self.dones[self.size % self.max_size] = done
        self.target[self.size % self.max_size] = target
        self.size += 1



with open(json_inputs, 'r') as f:
    datadata = json.load(f)

observations = []
states = []
actions = []
rewards = []
critic_rewards = []
next_states = []
dones = []
targets = []
dialogs = []

for item in datadata:
  # i would like to convert it to dialog
    observations.append(item['user_input'])
    states.append(item['state'])

    dialogs.append(item['state'])

    critic_rewards.append(item['critic_rewards'])


    actions.append(item['action'])
    rewards.append(item['actor_rewards'])

    if item.get("update_state", "") == "":
      next_states.append(item["state"])
    else:
      next_states.append(item["update_state"])
    # next_states.append(item['update_state'] if item['update_state'] else item['state'])  # Use 'state' if 'update_state' is empty
    dones.append(item['done'])
    targets.append(item['target'])

combined_states = [ob + state for ob, state in zip(observations, states)]
combined_next_states = [ob + next_state for ob, next_state in zip(observations, next_states)]


# Instantiate ReplayBuffer
replay_buffer = ReplayBuffer(batch_size=2, capacity=10000)

############## convert state->utter to state -> dialog ###############
# Insert data into ReplayBuffer
for observation, utter, dialog, action, reward, critic_reward, next_state, done, target in zip(combined_states, observations, dialogs, actions, rewards, critic_rewards, combined_next_states, dones, targets):
    replay_buffer.insert(
        state = observation,
        observation = utter,
        dialog = utter,
        action = action,
        reward=np.array(reward),
        critic_reward = np.array(critic_reward),
        next_state=next_state,
        done=np.array(done),
        target =np.array(target)
    )


import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Union, Tuple
from tqdm import tqdm
import torch.nn.functional as F
from transformers import TrainingArguments


class ArcherTrainingArguments(TrainingArguments):
    def __init__(self, critic_lr: float = 1e-4, lm_lr: float = 1e-4, gamma: float = 0.99, tau: float = 0.2,
                 actor_epochs: int = 4, critic_epochs: int = 4, updates_per_epoch: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.critic_lr = critic_lr
        self.lm_lr = lm_lr
        self.gamma = gamma
        self.tau = tau
        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs
        self.updates_per_epoch = updates_per_epoch


class DoubleCritic(nn.Module):
    def __init__(self, device, critic_lm, tokenizer, cache_dir, in_dim, out_dim):
        super(DoubleCritic, self).__init__()
        self.device = device
        mid_dim = 256

        self.base_lm = AutoModel.from_pretrained(critic_lm).to(device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(critic_lm)

        self.critic1 = nn.Sequential(nn.Linear(in_dim*2, in_dim), nn.ReLU(),
                                     nn.Linear(in_dim, in_dim), nn.ReLU(),
                                     nn.Linear(in_dim, out_dim)).to(device)

        self.critic2 = nn.Sequential(nn.Linear(in_dim*2, in_dim), nn.ReLU(),
                                     nn.Linear(in_dim, in_dim), nn.ReLU(),
                                     nn.Linear(in_dim, out_dim)).to(device)

        self.v_critic1 = nn.Sequential(nn.Linear(in_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, out_dim)).to(device)


    def forward(self, observation, action, detach_model=False):
        obs_ids = self.base_tokenizer(observation, padding=True, return_tensors='pt', max_length=512, truncation=True).to(self.device)
        action_ids = self.base_tokenizer(action, padding=True, return_tensors='pt', max_length=512, truncation=True).to(self.device)

        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).pooler_output
                action_states = self.base_lm(**action_ids).pooler_output
        else:
            lm_states = self.base_lm(**obs_ids).pooler_output
            action_states = self.base_lm(**action_ids).pooler_output

        q_states = torch.cat([lm_states, action_states], dim=1)
        return self.critic1(q_states), self.critic2(q_states), self.v_critic1(lm_states) #self.v_critic2(lm_states)


class ArcherAgent(nn.Module):
    def __init__(self, device, accelerator, lora_model, tokenizer, critic_lm="roberta-base", dropout=0.1,
                 critic_lr=1e-5, lm_lr=1e-5, gamma=0.99, tau=0.1):
        super(ArcherAgent, self).__init__()
        self.model = lora_model
        self.critic = DoubleCritic(device, critic_lm, tokenizer, cache_dir='~/.cache', in_dim=768, out_dim=1)
        self.target_critic = DoubleCritic(device, critic_lm, tokenizer, cache_dir='~/.cache', in_dim=768, out_dim=1)
        self.soft_update_target_critic(tau)
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.MSELoss()
        self.lm_optimizer = torch.optim.Adam(self.model.parameters(), lr=lm_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau

        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        ###############################################################


        self.model, self.critic, self.target_critic = self.accelerator.prepare(self.model, self.critic, self.target_critic)

        self.critic_optimizer, self.lm_optimizer = self.accelerator.prepare(self.critic_optimizer, self.lm_optimizer)



    def critic_loss(self, observation, dialog, state, action, reward, critic_reward, next_state, done, target, **kwargs):

        reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.model).device, dtype = self.accelerator.unwrap_model(self.model).dtype).flatten()
        done = torch.Tensor(done).to(self.accelerator.unwrap_model(self.model).device, dtype = self.accelerator.unwrap_model(self.model).dtype).flatten()
        critic_reward = torch.Tensor(critic_reward).to(self.accelerator.unwrap_model(self.model).device, dtype = self.accelerator.unwrap_model(self.model).dtype).flatten()

        q1, q2, v1 = self.critic(observation, state, detach_model=False)

    ####anoter better option#################
        alpha = 0.9 
        beta= 0.45 
        combined_reward = alpha * reward + beta * critic_reward

        with torch.no_grad():

            target_q1, target_q2, target_v1= self.target_critic(observation, next_state, detach_model=True)
            target_v1 = combined_reward + (1-done) * target_v1.flatten() * self.gamma

        q1, q2, v1 = q1.flatten(), q2.flatten(), v1.flatten()
        target_q1, target_q2 = target_q1.flatten(), target_q2.flatten()
        #this is for trianing the q-model
        q1_loss = self.criterion(q1, target_v1)
        q2_loss = self.criterion(q2, target_v1)
        #this is for training the value model
        v1_loss = self.criterion(v1, target_q1)
        v2_loss = self.criterion(v1, target_q2)

        # print("q1.loss:", q1_loss, "q2.loss:", q2_loss, "v1.loss:", v1_loss, "v2.loss:", v2_loss)

        return {"q1.loss": q1_loss, "q2.loss": q2_loss, "v1.loss": v1_loss, "v2.loss": v2_loss}


    def actor_loss(self, observation, advantage, state, **kwargs):
        log_prob = self.compute_nll_loss(observation, state)
        advantages = advantage.flatten()
     
        pg_loss = -torch.mean(log_prob.flatten()* advantages)

        print("pg----", pg_loss, advantages)
        return {"pg_loss": pg_loss}


    def compute_nll_loss(self, observation, state):

        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=150, truncation = True).to(self.device)
        action_ids = self.tokenizer(state, return_tensors='pt', padding=True, max_length=150, truncation = True).to(self.device)

        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
    
        attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
                                dim = 1)
        outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        values = None
        if isinstance(outputs, Tuple):
            values, outputs = outputs
        prediction_probs = self.softmax(outputs.logits)

        selected_prediction_probs = torch.take_along_dim(prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1],\
                                                 action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        if values is not None:
            return values[:, obs_ids["attention_mask"].size(1)-1:-1], torch.log(selected_prediction_probs)*action_ids["attention_mask"], action_ids["attention_mask"]
        else:
            # print(torch.log(selected_prediction_probs), action_ids["attention_mask"])
            return torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim = 1)


    def compute_nll_loss11(self, observation, state):
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=100, truncation=True).to(self.device)
        action_ids = self.tokenizer(state, return_tensors='pt', padding=True, max_length=150, truncation=True).to(self.device)

        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
    
        attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
                                dim = 1)
        outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)

        logits = outputs.logits

        # Separate the logits corresponding to the observation part
        # The observation starts after the length of the state (utterance)
        state_length = obs_ids["input_ids"].size(1)
        observation_logits = logits[:, state_length:, :]  # Shape: [batch_size, observation_length, vocab_size]

        observation_logits = observation_logits.reshape(-1, observation_logits.size(-1))  # [batch_size * observation_length, vocab_size]
        observation_labels = action_ids["input_ids"].reshape(-1)  # [batch_size * observation_length]

        # Compute cross-entropy loss
        loss = F.cross_entropy(observation_logits, observation_labels, ignore_index=self.tokenizer.pad_token_id)
        return loss


    def soft_update_target_critic(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def save(self, path):
        torch.save({'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
                    'critic_state_dict': self.accelerator.unwrap_model(self.critic).state_dict(),
                    'target_critic_state_dict': self.accelerator.unwrap_model(self.target_critic).state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'lm_optimizer_state_dict': self.lm_optimizer.state_dict()}, path)


class ArcherTrainer(Trainer):
    def __init__(self, agent, replay_buffer, *args, **kwargs):
        super().__init__(model=agent.model, *args, **kwargs)
        self.agent = agent
        self.replay_buffer = replay_buffer

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, **kwargs):
        total_updates = self.args.num_train_epochs * self.args.updates_per_epoch

        for update in tqdm(range(total_updates), desc="Training"):
            self.update_critic()
            self.update_actor()

            if (update + 1) % self.args.save_steps == 0:


                self.agent.save(f"model_checkpoint_{update+1}.pth")


    def update_critic(self):
        for _ in range(self.args.critic_epochs):
            data = [self.replay_buffer.sample(1) for _ in range(self.args.gradient_accumulation_steps * self.replay_buffer.batch_size)]
            for d in data:
                for k, v in d.items():
                    d[k] = v[0]
            dataloader = DataLoader(DummyDataset(data), batch_size=self.replay_buffer.batch_size)

            self.agent.critic_optimizer.zero_grad()
            for batch in dataloader:
                loss_dict = self.agent.critic_loss(**batch)
                loss = sum(loss for loss in loss_dict.values() if isinstance(loss, torch.Tensor))
                self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(self.agent.critic.parameters(), self.args.max_grad_norm)
            self.agent.critic_optimizer.step()
            self.agent.soft_update_target_critic(self.args.tau)

    def update_actor(self):
        for _ in range(self.args.actor_epochs):
            data = [self.replay_buffer.sample(1) for _ in range(self.args.gradient_accumulation_steps * self.replay_buffer.batch_size)]
            for d in data:
                for k, v in d.items():
                    d[k] = v[0]
            dataloader = DataLoader(DummyDataset(data), batch_size=self.replay_buffer.batch_size)

            dataloader = self.accelerator.prepare(dataloader)


            self.agent.lm_optimizer.zero_grad()

            for batch in dataloader:

                with torch.no_grad():
                    # reward_batch = batch["reward"]
                    # done_batch = batch["done"]
                    # reward_batch = torch.Tensor(reward_batch).to(self.accelerator.unwrap_model(self.model).device, dtype = self.accelerator.unwrap_model(self.model).dtype).flatten()
                    # done_batch = torch.Tensor(done_batch).to(self.accelerator.unwrap_model(self.model).device, dtype = self.accelerator.unwrap_model(self.model).dtype).flatten()
                    # reward_batch = reward_batch.unsqueeze(1)
                    # done_batch = done_batch.unsqueeze(1)

                    q1, q2, v = self.agent.critic(batch["observation"], batch["state"])
                    # q1, q2, v = self.agent.critic(batch["observation"], batch["next_state"])

                    # q1, q2, v1, v2 = self.critic(batch["next_state"], batch["target"])

                    # q11, q22, v11, v22 = self.agent.critic(batch["observation"], batch["next_state"]) ## in  value state, the true label has nothing to do with

                    q = torch.minimum(q1, q2)

                    advantages = q - v

                loss_dict = self.agent.actor_loss(advantage=advantages, state =batch["state"], observation = list(batch["observation"]))

                loss = loss_dict["pg_loss"]

                self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(self.agent.model.parameters(), self.args.max_grad_norm)
            self.agent.lm_optimizer.step()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = ArcherAgent(device, accelerator, lora_model, tokenizer)


# Fill the replay buffer here

training_args = ArcherTrainingArguments(
    output_dir="./results",
    num_train_epochs= 5,#6
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2, 
    # learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir='./logs',
    # logging_steps=3,
    save_steps=5,
    fp16=True,
    max_grad_norm=0.001,
    critic_lr=1e-4,
    lm_lr=1e-5,
    gamma=0.99,
    tau=0.1,
    actor_epochs = 2, 
    critic_epochs = 2, 
    updates_per_epoch=2
)


trainer = ArcherTrainer(
    agent=agent,
    args=training_args,
    replay_buffer=replay_buffer
)

trainer.train()
