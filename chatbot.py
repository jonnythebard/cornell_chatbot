import numpy as np
import tensorflow as tf
import re
import time

lines = open("movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
conversations = open("movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(","))

questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text


clean_questions = []
clean_answers = []

for question in questions:
    clean_questions.append(clean_text(question))
for answer in answers:
    clean_answers.append(clean_text(answer))


# word's number of occurences
word2count = {}
for sentence in clean_questions + clean_answers:
    for word in sentence.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# tokenization & thresholding
threshold = 20
questionwords2int = {}
answerwords2int = {}
tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]

word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        # make two separate dictionaries in case we want to use different thresholds
        # to filter out the non frequent words in the dictionaries of the questions and the answers.
        questionwords2int[word] = word_number
        answerwords2int[word] = word_number
        word_number += 1

for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1
    answerwords2int[token] = len(answerwords2int) + 1

answersints2word = {_word_number: word for word, _word_number in answerwords2int.items()}

for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"

questions_into_int = []
answers_into_int = []

for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int["<OUT>"])
        else:
            ints.append(questionwords2int[word])
    questions_into_int.append(ints)

for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerwords2int:
            ints.append(answerwords2int["<OUT>"])
        else:
            ints.append(answerwords2int[word])
    answers_into_int.append(ints)

sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 26):
    for idx, value in enumerate(questions_into_int):
        if len(value) == length:
            sorted_clean_questions.append(questions_into_int[idx])
            sorted_clean_answers.append(answers_into_int[idx])


def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="target")
    lr = tf.placeholder(tf.float32, [None, None], name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, [None, None], name="keep_prob")  # dropout rate
    return inputs, targets, lr, keep_prob


epochs = 100
batch_size = 32
rnn_size = 1024
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

tf.reset_default_graph()
session = tf.InteractiveSession()
inputs, targets, lr, keep_prob = model_inputs()
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')
input_shape = tf.shape(inputs)
