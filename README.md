# LLM - Prompt Game

## Game description

The goal of this game is to generate prompts that increases the probability that the Large Language Model ([GPT-2](https://github.com/openai/gpt-2)) outputs a certain goal word next.

For example, if the goal word is 'fries' you could write 'I like to eat french' and GPT-2 might predict 'fries'.

Try to write a sequence of words that is as short as possible, but still makes GPT-2 predict the goal word.

You will get more points for shorter sequences, and for higher probability of the goal word.

The only rule is that you can't use the goal word in your input sequence.

## Keep in mind

Don't take the score too seriously, we are using a small version of GPT-2 released in 2019 so you might have had a great idea but the model was simply behaving strangely compared to the modern massive language models. That said, from personal experience it happens more often that you simly have bias in your own thinking and miss other possible continuations. This is what I thought was interesting and made the game extra fun, so it will also generate 5 possible continuations so that you can see for yourself what it writes instead when your goal word has a low probability.

## Installation

Tested with python 3.10.10

> py -m pip install -r requirements.txt

(Just torch and transformers)

Run src\game.py to start the game.

## Enjoy!