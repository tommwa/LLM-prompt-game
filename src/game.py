import os
import random

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def gen_rand_words():
    random_words = []
    folder = "data"
    file = "enwiki-2023-04-13-top1000.txt"
    with open(os.path.join(folder, file), "r") as f:
        for row in f.readlines():
            words = row.split(" ")
            random_words.append(words[0])
    random.shuffle(random_words)
    for i in range(len(random_words)):
        yield random_words[i]


def predict_next_word(model, tokenizer, input_text, goal_word):
    # Run GPT-2 on the input text.
    input_tokens = tokenizer(input_text, return_tensors="pt").input_ids
    generated_tokens = model.generate(
        input_tokens,
        do_sample=True,
        num_return_sequences=5,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=20,
        return_dict_in_generate=True,
    )

    # Print out the generated continuations.
    sequences = generated_tokens.sequences[:, input_tokens.shape[-1] :]
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    print("Some example continuations:")
    for text in texts:
        print(text)
        print("-----")

    # Find the probability of the goal word.
    probs = torch.stack(generated_tokens.scores, dim=1).softmax(-1)
    tok = tokenizer.encode(" " + goal_word)
    prob = probs[0][0][tok][0].item()
    return prob


# Prapare tokenizer and model.
print("Loading GPT-2 model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


# Introduce the game
print(
    "The goal of this game is to generate prompts that increases the probability that the Large Language Model (GPT-2) outputs a certain goal-word next.\
      \n\nFor example, if the goal word is 'fries' you could write 'I like to eat french' and GPT-2 might predict 'fries'.\
      \nTry to write a sequence of words that is as short as possible, but still makes GPT-2 predict the goal word.\
      \nYou will get more points for shorter sequences, and for higher probability of the goal word.\n \
      \nThe only rule is that you can't use the goal word in your input sequence.\
      \n"
)

# Prepare word generator
print(
    "Do you want to seed the game to fix order of the words or do you want random words? (Only recomended to fix it if you want to compare score with a friend, if you want to play multiple times random is preferred.)"
)
to_seed = None
while to_seed not in ["fix", "random", "f", "r"]:
    to_seed = input("Type 'f' or 'r' for fix or random: ")
if to_seed in ["fix", "f"]:
    random.seed(0)
word_generator = gen_rand_words()

# Game loop.
scores = []
for goal in word_generator:
    # It's a bit cumbersome the probability of the next word if it consists of multiple tokens so we simply skip them.
    tok = tokenizer.encode(" " + goal)
    if len(tok) > 1:
        continue

    # Ask user for input sequence.
    user_wants_to_quit = False
    while True:
        user_inp = input(
            f"(Enter 'q' to quit)\nThe goal continuation word is:\n{goal}\n"
        )
        if user_inp == "q":
            user_wants_to_quit = True
        user_inp = user_inp.strip()
        input_len = len(user_inp.split(" "))
        if not user_inp:
            print("Input sequence must be non-empty. Try again!")
            continue
        if goal in user_inp:
            print("You can't use the goal word in your input sequence. Try again")
            continue
        break
    if user_wants_to_quit:
        break

    # Use input to generate 5 example continuations and find goal probability.
    prob = predict_next_word(model, tokenizer, input_text=user_inp, goal_word=goal)

    # Give a score based on the probability of the goal word and the length of the input sequence.
    score = prob / input_len * 1000
    scores.append(score)
    average_score = sum(scores) / len(scores)

    print(
        f"You input {input_len} number of words and got the probability {prob:.2f} which gives you {prob:.2f} / {input_len} * 1000 = {score:.2f}."
    )
    print(f"Your average score is {average_score}.\n")
    print("---------------------------------------------------------\n")
