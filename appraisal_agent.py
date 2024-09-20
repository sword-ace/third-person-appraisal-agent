# -*- coding: utf-8 -*-

import json
import numpy as np
import string
from tqdm import tqdm

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

# def EM(prediction, sentiment) -> bool:
#     return prediction.lower() == sentiment.lower()


# # # # # # # Initialize agents with the DataFrame
# agents = [PredictAgent(row['ticker'], row['summary'], row['target'], predict_llm=model, tokenizer=tokenizer) for _, row in data.iterrows()]
# print("Loaded Train Agents.")

# # Run agents
# for agent in agents:
#     agent.run()
#     if agent.is_correct():
#         print(agent.prediction, agent.target)
#         print(f"Correct prediction for {agent.ticker}")

#     else:
#         print(agent.prediction, agent.target)
#         print(f"Incorrect prediction for {agent.ticker}")

"""## agent design ##"""

# class PredictAgent2:
#                  ticker: str,
#                  summary: str,
#                  target: int,
#                  predict_llm = model
#                  ) -> None:

#         self.ticker = ticker
#         self.summary = summary
#         self.target = target
#         self.prediction = ''

#         self.predict_prompt = PREDICT_INSTRUCTION
#         self.predict_examples = PREDICT_EXAMPLES
#         self.llm = predict_llm

#         self.__reset_agent()

#     def run(self, reset=True) -> None:
#         if reset:
#             self.__reset_agent()

#         facts = "Facts:\n" + self.summary + "\n\nEmotion Label: "
#         self.scratchpad += facts
#         print(facts, end="")

#         self.scratchpad += self.prompt_agent()
#         response = self.scratchpad.split('Emotion Label: ')[-1]
#         self.prediction = response.split()[0]
#         print(response, end="\n\n\n\n")

#         self.finished = True

#     def prompt_agent(self) :
#         return self.llm(self._build_agent_prompt())

#     def _build_agent_prompt(self) -> str:
#         return self.predict_prompt.format(
#                             ticker = self.ticker,
#                             examples = self.predict_examples,
#                             summary = self.summary)

#     def is_finished(self) -> bool:
#         return self.finished

#     def is_correct(self) -> bool:
#         return EM(self.target, self.prediction)

#     def __reset_agent(self) -> None:
#         self.finished = False
#         self.scratchpad: str = ''

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

# # # # Load model
# model2, tokenizer2 = FastLanguageModel.from_pretrained(
#     model_name="unsloth/gemma-7b-bnb-4bit",
#     max_seq_length=max_seq_length,
#     dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
#     load_in_4bit=True, # Use 4bit quantization to reduce memory usage. Can be False

#       # This could be customized if more fine-grained control is needed
# #     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )


# # ############################## Do model patching and add fast LoRA weights###################################
# model2 = FastLanguageModel.get_peft_model(
# model2,
# r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
# target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                   "gate_proj", "up_proj", "down_proj",],
# lora_alpha = 16,
# lora_dropout = 0, # Supports any, but = 0 is optimized
# bias = "none",    # Supports any, but = "none" is optimized

# # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
# use_gradient_checkpointing = True, # True or "unsloth" for very long context
# random_state = 3407,
# use_rslora = False,  # We support rank stabilized LoRA
# loftq_config = None, # And LoftQ
# )



"""## Test for generating efficient reflections ##"""

# class ReflectionGenerator:
#         self.reflect_llm = reflect_llm
#         self.tokenizer = tokenizer

#     def generate_reflection(self, previous_label: str, context: str, previous_trial: str) -> str:
#         # prompt = (
#         #     f"You will be given a previous reasoning trial in which you were given access to a list of facts to predict the emotion label of the last utterance of {context}"
#         #     f"You were unsuccessful in tackling the task because you gave the wrong emotion label of {previous_label}. "
#         #     f"Please re-evaluate the emotional context and overall tone of the last utterance.Use complete sentences.\n\n"
#         #     f"Incorrect Label:\n{previous_label}\n\n"
#         #     f"Previous trial:\nFacts:\n{previous_trial}\n\n"
#         #     f"Relfection:\n"
#         # )

#         prompt = (
#             f"You will be given a previous trial in which you were given access to a list of facts to predict the emotion label of the last utterance of {context}"
#             f"You were unsuccessful in deductive reasoning because you gave the wrong emotion label of {previous_label}. "
#             f"Re-evaluate the emotional context and overall tone of the last utterance. Use complete sentences.\n\n"
#             # f"Incorrect Label:\n{previous_label}\n\n"
#             # f"Previous trial:\nFacts:\n{previous_trial}\n\n"
#             f"Relfection:\n"
#         )
#         # inputs = self.tokenizer.encode(prompt, return_tensors="pt")
#         encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

#         outputs = self.reflect_llm.generate(
#         input_ids=encoding.input_ids,
#         attention_mask=encoding.attention_mask,
#         pad_token_id=tokenizer.eos_token_id,
#         # max_new_tokens = 120,
#         # repetition_penalty=1.2,
#         max_length=encoding.input_ids.size(1) + 50,  # Extend max_length by 150 tokens for the generated summary
#         num_return_sequences=1,
#         use_cache=True

#        )

#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response.strip()


# previous_label = "happy"
# context = "No, that desire will fade along with our passion."
# previous_trial = (
#     "Contextualization: The last utterance is a response to the speaker's previous statement about the dangers of being serious and the need to remain lighthearted. "
#     "The speaker is trying to convince the other speaker that they should not take things too seriously and should try to maintain a playful and carefree attitude.\n"
#     "Emotional tone intensity: The emotional tone intensity of the last utterance is relatively low, around a 3 or 4 out of 10. "
# )

# # Instantiate the ReflectionGenerator
# reflection_generator = ReflectionGenerator(model2, tokenizer2)
# # Generate the reflection
# reflection = reflection_generator.generate_reflection(previous_label, context, previous_trial)
# print(reflection)



"""## this is the test case for generating new emotional label ##"""


# class EmotionLabelGenerator:
#         self.predict_llm = predict_llm
#         self.tokenizer = tokenizer

#     def generate_emotion_label(self, context: str, facts: str, reflections: str, examples: str, previous_label: str) -> str:
#         prompt = f"""
#             #### Instruction:
#             Given a list of facts and reflections, predict the emotion for the last utterance of {context}. Give your response in this format:
#             (1) Emotion Label: Choose from the following set: ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
#             (2) Deduction: Explain your reasoning by clearly stating the key facts and how they logically lead to the predicted emotion.

#             ### Examples:
#             {examples}
#             (END OF EXAMPLES)

#             ### Facts:
#             {facts}

#             ### Reflections:
#             {reflections}

#             ### Emotion Label:"""

#         # inputs = self.tokenizer.encode(prompt, return_tensors="pt")
#         encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

#         outputs = self.predict_llm.generate(
#         input_ids=encoding.input_ids,
#         attention_mask=encoding.attention_mask,
#         pad_token_id=tokenizer.eos_token_id,
#         # max_new_tokens = 120,
#         # repetition_penalty=1.2,
#         max_length=encoding.input_ids.size(1) + 150,  # Extend max_length by 150 tokens for the generated summary
#         num_return_sequences=1,
#         use_cache=True

#           )
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response.strip()


# # Instantiate the EmotionLabelGenerator
# emotion_label_generator = EmotionLabelGenerator(model2, tokenizer2)

# # Define the ticker, facts, reflections, and examples
# previous_label = "happy"
# context = "No, that desire will fade along with our passion."
# previous_trial = (
#     "Contextualization: The last utterance is a response to the speaker's previous statement about the dangers of being serious and the need to remain lighthearted. "
#     "The speaker is trying to convince the other speaker that they should not take things too seriously and should try to maintain a playful and carefree attitude.\n"
#     "Emotional tone intensity: The emotional tone intensity of the last utterance is relatively low, around a 3 or 4 out of 10. "
# )

# facts = (
#     "Contextualization: The last utterance is a response to the speaker's previous statement about the dangers of being serious and the need to remain lighthearted. "
#     "The speaker is trying to convince the other speaker that they should not take things too seriously and should try to maintain a playful and carefree attitude.\n"
#     "Emotional tone intensity: The emotional tone intensity of the last utterance is relatively low, around a 3 or 4 out of 10."
# )

# reflections = (
#     "The last utterance in the trial is ""No, that desire will fade along with our passion."" The emotional context of this utterance is that the speaker is feeling sad and disappointed. The overall tone of the utterance is one of sadness and regret."
# )

# examples = PREDICT_EXAMPLES
# # Generate the emotion label
# emotion_label = emotion_label_generator.generate_emotion_label(context, facts, reflections, examples, previous_label)
# # print("hahaha", emotion_label)

# kk= emotion_label.split('Emotion Label:')[-1]

# print(kk)

# #### Instruction:
#             Given a list of facts and reflections, predict the emotion for the last utterance of No, that desire will fade along with our passion.. Give your response in this format:
#             (1) Emotion Label: Choose from the following set: ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
#             (2) Deduction: Explain your reasoning by clearly stating the key facts and how they logically lead to the predicted emotion.
#             ### Examples:

# ### Facts:
# Contextualization: In the last utterance, "I know," Speaker1 is acknowledging Speaker2's comment about the difficulty of leaving a new baby behind and expressing understanding.
# Emotional Tone: The emotional tone of the utterance "I know" is empathetic and resigned, as Speaker1 is showing understanding and agreement with the hardship expressed by Speaker2.
# Emotional Tone Intensity: The emotional tone intensity is moderate to high, reflecting a significant level of empathy and resignation due to the context of discussing sacrifices and the difficulty of leaving a new baby.
# ### Emotion Label:
# sad
# ### Deduction:
# The context and tone suggest that Speaker1 is feeling the weight of the situation and empathizes deeply with the challenges mentioned. The resigned and understanding tone, combined with the significant emotional intensity, logically leads to the emotion of sadness. This emotion reflects the recognition of a tough and unfortunate reality that both speakers are facing.

#             (END OF EXAMPLES)
#             ### Facts
#             Contextualization: The last utterance is a response to the speaker's previous statement about the dangers of being serious and the need to remain lighthearted. The speaker is trying to convince the other speaker that they should not take things too seriously and should try to maintain a playful and carefree attitude.
# Emotional tone intensity: The emotional tone intensity of the last utterance is relatively low, around a 3 or 4 out of 10.
#             ### Reflections
#             The last utterance in the trial is No, that desire will fade along with our passion. The emotional context of this utterance is that the speaker is feeling sad and disappointed. The overall tone of the utterance is one of sadness and regret.
#             ### Emotion Label:
#             sad
#             ### Deduction:
#             The last utterance in the trial is No, that desire will fade along with our passion. The emotional context of this utterance is that the speaker is feeling sad and disappointed. The overall tone of the utterance is one of sadness and regret.


# class PredictReflectAgent(PredictAgent):
#                  ticker: str,
#                  summary: str,
#                  target: str,
#                  predict_llm = model,
#                  reflect_llm = model2,
#                  tokenizer = tokenizer,
#                  tokenizer2 = tokenizer2
#                  ) -> None:

#         self.predict_llm = predict_llm
#         self.reflect_llm = reflect_llm
#         self.reflect_prompt = REFLECT_INSTRUCTION
#         self.agent_prompt = PREDICT_REFLECT_INSTRUCTION
#         self.reflections = []

#         self.tokenizer = tokenizer2
#         self.reflections_str: str = ''

#     def run(self, reset=True) -> None:
#         if self.is_finished() and not self.is_correct():
#             self.reflect()

#         PredictAgent.run(self, reset=reset)


#     def reflect(self) -> None:
#         print('Reflecting...\n')
#         reflection = self.prompt_reflection()
#         self.reflections += [reflection]
#         self.reflections_str = format_reflections(self.reflections)
#         print("reflection format:")
#         print(self.reflections_str, end="\n\n\n\n")

#     def prompt_reflection(self) -> str:

#         prompt = self._build_reflection_prompt()

#         print("prompt-----refle----", prompt)
#         print("scrach-----", self.scratchpad)


#         # inputs = self.tokenizer.encode(prompt, return_tensors="pt")

#         # model_output = model2.generate(inputs, max_new_tokens = 128, use_cache = True)



#         encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)


#         model_output = model2.generate(
#                 input_ids=encoding.input_ids,
#                 attention_mask=encoding.attention_mask,
#                 pad_token_id=self.tokenizer.eos_token_id,
#                 # max_new_tokens = 120,
#                 # repetition_penalty=1.2,
#                 max_length=encoding.input_ids.size(1) + 50,  # Extend max_length by 150 tokens for the generated summary
#                 num_return_sequences=1,
#                 use_cache=True

#             )


#         # Decode the model output
#         if isinstance(model_output, torch.Tensor):
#             model_output = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
#         elif isinstance(model_output, list):
#             model_output = ' '.join(map(str, model_output))

#         return model_output

#         # if isinstance(model_output, torch.Tensor):
#         #     model_output = model_output.tolist()  # or `.numpy()` followed by `.tolist()` if it's a numpy array
#         # if isinstance(model_output, list):
#         #     model_output = ' '.join(map(str, model_output))
#         # return model_output

#     def _build_reflection_prompt(self) -> str:

#         parsed_response = self.scratchpad.split('Emotion Label:')[-1]
#         # print("showwwww-----------------", parsed_response)
#         prev_pred = parsed_response.split()[0]
#         print("showwwww-----------------", prev_pred)

#         # return self.reflect_prompt.format(ticker = self.ticker)

#         return self.reflect_prompt.format(
#                             ticker = self.ticker,
#                             labs = prev_pred

#                             )

#     def _build_agent_prompt(self) -> str:
#         prompt = self.agent_prompt.format(
#                             ticker = self.ticker,
#                             examples = self.predict_examples,
#                             reflections = self.reflections_str,
#                             summary = self.summary)

#         # prompt = self.agent_prompt.format(
#         #                     ticker = self.ticker,

#         #                     reflections = self.reflections_str,
#         #                     summary = self.summary)

#         return prompt

#     def run_n_shots(self, model, tokenizer, reward_model, num_shots=4, reset=True) -> None:
#         self.llm = NShotLLM(model, tokenizer, reward_model, num_shots)
#         PredictAgent.run(self, reset=reset)


# def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
#     if reflections == []:
#         return ''
#     else:
#         return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# agent_cls = PredictReflectAgent
# agents = [agent_cls(row['ticker'], row['summary'], row['target'], model, model2, tokenizer, tokenizer2) for _, row in data.iterrows()]
# print("Loaded Train Agents.")

# num_reflect_trials = 2
# # # Train supervised policy
# # supervised_finetune()
# # merge_peft_adapter(model_name= output_path, output_name= rl_base_model)
# datasets_dir = "./datasets/"
# ## if only has 1 sample will face the shape problem? ##
# # Collect comparison data
# comparison_data = []

# for trial in range(num_reflect_trials):
#     ## for agents only return false
#     for idx, agent in enumerate([a for a in agents if not a.is_correct()]):

#         prev_response = agent.scratchpad.split('Emotion Label: ')[-1]

#         agent.run()


#         if agent.is_correct():
#             print("agent is correcting the mind..----------")
#             print(agent._build_agent_prompt(), "\n\n\n")
#             prompt = remove_reflections(agent._build_agent_prompt())
#             response = agent.scratchpad.split('Emotion Label: ')[-1]

#             sample = {"user_input": prompt, "completion_a": prev_response, "completion_b": response}
#             # print("show-------------", sample)
#             comparison_data.append(sample)

#     correct, incorrect = summarize_trial(agents)
#     print(f'Finished Trial {trial+1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

# os.makedirs(datasets_dir, exist_ok=True)
# comparison_data_path = os.path.join(datasets_dir, "comparison_data.json")

# if comparison_data:
#     with open(comparison_data_path, 'w') as f:
#         f.write(json.dumps(comparison_data))




# class PredictAgent:
#         self.ticker = ticker
#         self.summary = summary
#         self.target = target
#         self.predict_llm = predict_llm

#     def run(self, reset: bool = True) -> None:
#         pass

#     def is_finished(self) -> bool:
#         return False

#     def is_correct(self) -> bool:
#         return False

# class PredictReflectAgent(PredictAgent):
#                  ticker: str,
#                  summary: str,
#                  target: str,
#                  predict_llm: Any,
#                  reflect_llm: Any,
#                  tokenizer: Any
#                  ) -> None:
#         self.reflect_llm = reflect_llm
#         self.reflect_prompt = REFLECT_INSTRUCTION
#         self.agent_prompt = PREDICT_REFLECT_INSTRUCTION
#         self.reflections = []
#         self.tokenizer = tokenizer
#         self.reflections_str: str = ''
#         self.scratchpad = ''  # Ensure scratchpad is initialized


#     def run(self, reset=True) -> None:
#         if self.is_finished() and not self.is_correct():
#             self.reflect()
#         super().run(reset=reset)

#     def reflect(self) -> None:
#         print('Reflecting...\n')
#         reflection = self.prompt_reflection()
#         self.reflections.append(reflection)
#         self.reflections_str = format_reflections(self.reflections)
#         print(self.reflections_str, end="\n\n\n\n")

#     def prompt_reflection(self) -> str:
#         prompt = self._build_reflection_prompt()
#         inputs = self.tokenizer.encode(prompt, return_tensors="pt")
#         model_output = self.reflect_llm.generate(inputs, max_new_tokens=128, use_cache=True)

#         if isinstance(model_output, torch.Tensor):
#             model_output = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
#         elif isinstance(model_output, list):
#             model_output = ' '.join(map(str, model_output))

#         return model_output

#     def _build_reflection_prompt(self) -> str:
#         return self.reflect_prompt.format(ticker=self.ticker, scratchpad=self.scratchpad)

#     def _build_agent_prompt(self) -> str:
#         return self.agent_prompt.format(
#             ticker=self.ticker,
#             examples='',  # assuming predict_examples is defined elsewhere
#             reflections=self.reflections_str,
#             summary=self.summary
#         )

#     def run_n_shots(self, model: Any, tokenizer: Any, reward_model: Any, num_shots=4, reset=True) -> None:
#         self.llm = NShotLLM(model, tokenizer, reward_model, num_shots)
#         super().run(reset=reset)


# def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
#     if not reflections:
#         return ''
#     return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# def EM(prediction: str, sentiment: str) -> bool:
#     return prediction.lower() == sentiment.lower()

# # PREDICT_REFLECT_INSTRUCTION = """ Given a list of facts and refelction, predict the emotion behind the last utterance of {ticker}. Give your response in this format:

# # (1) Emotion Label: Choose from the following set: ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
# # (2) Explanation: Provide a single, short paragraph explaining your reasoning.

# # Here are some examples:
# # {examples}
# # (END OF EXAMPLES)

# # {reflections}

# # Emotion Label:"""


# PREDICT_REFLECT_INSTRUCTION ="""
# ### Instruction:
# Given a list of facts and refelctions, predict the emotion for the last utterance of {ticker}. Give your response in this format:
# (1) Emotion Label: Choose from the following set: ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
# (2) Deduction: Explain your reasoning by clearly stating the key facts and how they logically lead to the predicted emotion.

# ### Examples:
# {examples}
# (END OF EXAMPLES)

# ### Facts:
# {summary}

# ### Reflections:
# {reflections}

# ### Emotion Label:"""


# # #### Instruction:
# #             Given a list of facts and reflections, predict the emotion for the last utterance of {context}. Give your response in this format:
# #             (1) Emotion Label: Choose from the following set: ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
# #             (2) Deduction: Explain your reasoning by clearly stating the key facts and how they logically lead to the predicted emotion.
# #             ### Examples:
# #             {examples}
# #             (END OF EXAMPLES)
# #             ### Facts
# #             {facts}
# #             ### Reflections
# #             {reflections}
# #             ### Emotion Label:"""


# #  prompt = (
# #             f"You will be given a previous trial in which you were given access to a list of facts to predict the emotion label of the last utterance of {context}"
# #             f"You were unsuccessful in deductive reasoning because you gave the wrong emotion label of {previous_label}. "
# #             f"Re-evaluate the emotional context and overall tone of the last utterance. Use complete sentences.\n\n"
# #             # f"Incorrect Label:\n{previous_label}\n\n"
# #             # f"Previous trial:\nFacts:\n{previous_trial}\n\n"
# #             f"Relfection:\n"
# #         )


# REFLECTION_HEADER = 'You have attempted to tackle the following task before and failed.'


# # 'You have attempted to tackle the following task before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly tackling the given task.\n'

# # REFLECT_INSTRUCTION = """Please re-evaluate the emotional context and overall tone of the last utterance {ticker}.
# # Previous trial:
# # {scratchpad}

# # Reflections:"""

# # REFLECT_INSTRUCTION = """
# # #### Instruction
# # You will be given a previous trial in which you were given access to a list of facts to predict the emotion label of the last utterance of {ticker}. You were unsuccessful in tackling the task because you gave the wrong emotion label. Please re-evaluate the emotional context and overall tone of the last utterance. Use complete sentences.

# # ### Relfection:
# # """


# REFLECT_INSTRUCTION ="""
# ### Instruction:
# You will be given a previous trial in which you were given access to a list of facts to predict the emotion label of the last utterance of {ticker}. You were unsuccessful in tackling the task because you gave the wrong emotion label of {labs}. Please re-evaluate the emotional context and overall tone of the last utterance. Use complete sentences.

# ### Relfection:"""


# # REFLECT_INSTRUCTION = """In a few sentences, reflect on the likely reasons for failure and the cognitive distortions or biases involved. Discuss methods for evaluating evidence objectively and techniques for incorporating mindfulness and rational analysis into decision-making.Use these strategies to prevent similar failures in the future. Please use complete sentences.
# # Previous trial:
# # {scratchpad}

# # Reflections:"""

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








"""