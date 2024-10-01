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



lora_model = get_peft_model(model, lora_config)
print("Using LoRA")
lora_model.print_trainable_parameters()

# Assuming the path to your dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import re

from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
device = accelerator.device



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


##########this part is for acotr-critic RL #########################

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Union, Tuple
from tqdm import tqdm
import torch.nn.functional as F
from transformers import TrainingArguments


class ReflectiveACArguments(TrainingArguments):
    def __init__(self, critic_lr: float = 1e-5, lm_lr: float = 1e-5, gamma: float = 0.99, tau: float = 0.2,
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

class AppraisalAgent(nn.Module):
    def __init__(self, device, accelerator, lora_model, tokenizer, critic_lm="roberta-base", dropout=0.1,
                 critic_lr=1e-5, lm_lr=1e-5, gamma=0.99, tau=0.1):
        super(AppraisalAgent, self).__init__()
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


        # ################
        # self.lm_optimizer = torch.optim.Adam(self.model.parameters(), lr = lm_lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)
        # ################


        self.model, self.critic, self.target_critic = self.accelerator.prepare(self.model, self.critic, self.target_critic)

        self.critic_optimizer, self.lm_optimizer = self.accelerator.prepare(self.critic_optimizer, self.lm_optimizer)



    def critic_loss(self, observation, dialog, state, action, reward, critic_reward, next_state, done, target, **kwargs):

        reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.model).device, dtype = self.accelerator.unwrap_model(self.model).dtype).flatten()
        done = torch.Tensor(done).to(self.accelerator.unwrap_model(self.model).device, dtype = self.accelerator.unwrap_model(self.model).dtype).flatten()
        critic_reward = torch.Tensor(critic_reward).to(self.accelerator.unwrap_model(self.model).device, dtype = self.accelerator.unwrap_model(self.model).dtype).flatten()

        q1, q2, v1 = self.critic(observation, state, detach_model=False)


    ####anoter better option#################
        alpha = 0.9 #0.8 #0.99
        beta= 0.45 #0.45 #0.45 #0.5 #0.45

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

        print("q1.loss:", q1_loss, "q2.loss:", q2_loss, "v1.loss:", v1_loss, "v2.loss:", v2_loss)

        return {"q1.loss": q1_loss, "q2.loss": q2_loss, "v1.loss": v1_loss, "v2.loss": v2_loss}


    def actor_loss(self, observation, advantage, state, **kwargs):
        log_prob = self.compute_nll_loss(observation, state)
        advantages = advantage.flatten()
 
        pg_loss = -torch.mean(log_prob.flatten()* advantages)


        # print("pg----", pg_loss.shape, log_prob.size, advantages)
        print("pg----", pg_loss, advantages)
        return {"pg_loss": pg_loss}


    def compute_nll_loss(self, observation, state):

        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=150, truncation = True).to(self.device)
        action_ids = self.tokenizer(state, return_tensors='pt', padding=True, max_length=150, truncation = True).to(self.device)

        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
        # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
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


class ReflectiveACTrainer(Trainer):
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
        
                    q1, q2, v = self.agent.critic(batch["observation"], batch["state"])
                   

                    q = torch.minimum(q1, q2)

                    advantages = q - v


                loss_dict = self.agent.actor_loss(advantage=advantages, state =batch["state"], observation = list(batch["observation"]))

                loss = loss_dict["pg_loss"]

                self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(self.agent.model.parameters(), self.args.max_grad_norm)
            self.agent.lm_optimizer.step()



# Usage example
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume you have defined lora_model, tokenizer, accelerator, and ReplayBuffer
agent = AppraisalAgent(device, accelerator, lora_model, tokenizer)

# Fill the replay buffer here

training_args = ReflectiveACArguments(
    output_dir="./results",
    num_train_epochs= 5,#6
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2, #2
    max_grad_norm=0.001,
    critic_lr=1e-5,
    lm_lr=1e-5,
    gamma=0.99,
    tau=0.1,
    actor_epochs = 2, #2 #3, #4
    critic_epochs = 2, #2 #3, #4
    updates_per_epoch=2
)


trainer = ReflectiveACTrainer(
    agent=agent,
    args=training_args,
    replay_buffer=replay_buffer
)

trainer.train()
