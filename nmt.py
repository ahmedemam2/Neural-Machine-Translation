import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Setting this env variable prevents TF warnings from showing up

import numpy as np
import tensorflow as tf
from collections import Counter
from utils import (sentences, train_data, val_data, english_vectorizer, portuguese_vectorizer, 
                   masked_loss, masked_acc, tokens_to_text)



portuguese_sentences, english_sentences = sentences
VOCAB_SIZE = 12000
UNITS = 256

# Size of the vocabulary
vocab_size_por = portuguese_vectorizer.vocabulary_size()
vocab_size_eng = english_vectorizer.vocabulary_size()


# This helps you convert from words to ids
word_to_id = tf.keras.layers.StringLookup(
    vocabulary=portuguese_vectorizer.get_vocabulary(), 
    mask_token="", 
    oov_token="[UNK]"
)

# This helps you convert from ids to words
id_to_word = tf.keras.layers.StringLookup(
    vocabulary=portuguese_vectorizer.get_vocabulary(),
    mask_token="", #marks the padding token used to ensure that sequences are of the same length
    oov_token="[UNK]",
    invert=True,
)    

unk_id = word_to_id("[UNK]")
sos_id = word_to_id("[SOS]")
eos_id = word_to_id("[EOS]")




class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):
        super(Encoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(  
            input_dim=vocab_size,
            output_dim=units,
            mask_zero=True
        )  
        self.rnn = tf.keras.layers.Bidirectional(  
            merge_mode="sum",  
            layer=tf.keras.layers.LSTM(
                units=units,
                return_sequences=True
            ),  
        )  

    def call(self, context):
        x = self.embedding(context)
        x = self.rnn(x)
        return x
    
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units):

        super().__init__()


        self.mha = ( 
            tf.keras.layers.MultiHeadAttention(
                key_dim=units,
                num_heads=1
            ) 
        )  

        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, context, target):
        attn_output = self.mha(
            query=target,
            value=context
        )  


        x = self.add([target, attn_output])

        x = self.layernorm(x)

        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):

        super(Decoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,  
            output_dim=units,  # Embedding dimension
            mask_zero=True  # Masks padding tokens (0)
        )  
        self.pre_attention_rnn = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True,
            return_state=True
        )  
        self.attention = CrossAttention(units)

        self.post_attention_rnn = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True
        )  
        self.output_layer = tf.keras.layers.Dense(
            units=vocab_size,  # Same as vocabulary size for word predictions
            activation=tf.nn.log_softmax  # Log softmax for stable probabilities
        )  

    def call(self, context, target, state=None, return_state=False):

        x = self.embedding(target)
        x, hidden_state, cell_state = self.pre_attention_rnn(x, initial_state=state)
        x = self.attention(context, x)
        x = self.post_attention_rnn(x)
        logits = self.output_layer(x)

        if return_state:
            return logits, [hidden_state, cell_state]

        return logits

class Translator(tf.keras.Model):
    def __init__(self, vocab_size, units):
        super().__init__()

        self.encoder = Encoder(vocab_size, units)
        self.decoder = Decoder(vocab_size, units)


    def call(self, inputs):

        context, target = inputs
        encoded_context = self.encoder(context)
        logits = self.decoder(encoded_context, target)
 

        return logits


def compile_and_train(model, epochs=20, steps_per_epoch=500):
    model.compile(optimizer="adam", loss=masked_loss, metrics=[masked_acc, masked_loss])

    history = model.fit(
        train_data.repeat(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
    )

    return model, history

def generate_next_token(decoder, context, next_token, done, state, temperature=0.0):

    logits, state = decoder(context, next_token, state=state, return_state=True)
    logits = logits[:, -1, :]
    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits / temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    logits = tf.squeeze(logits)
    next_token = tf.squeeze(next_token)
    logit = logits[next_token].numpy()
    next_token = tf.reshape(next_token, shape=(1,1))
    if next_token == eos_id:
        done = True
    
    return next_token, logit, state, done

def translate(model, text, max_length=50, temperature=0.0):

    tokens, logits = [], []
    text_tensor = tf.convert_to_tensor([text])
    context = english_vectorizer(text_tensor).to_tensor()
    context = model.encoder(context)
    next_token = tf.fill((1, 1), sos_id)
    state = [tf.zeros((1, UNITS)), tf.zeros((1, UNITS))]
    done = False
    
    for _ in range(max_length):

        try:
            next_token, logit, state, done = generate_next_token(
                decoder=model.decoder,
                context=context,
                next_token=next_token,
                done=done,
                state=state,
                temperature=temperature
            )
        except:
            raise Exception("Problem generating the next token")

        if done:
            break

        tokens.append(next_token)
        logits.append(logit)

    tokens = tf.concat(tokens, axis=-1)
    translation = tf.squeeze(tokens_to_text(tokens, id_to_word))
    translation = translation.numpy().decode()

    return translation, logits[-1], tokens


def generate_samples(model, text, n_samples=4, temperature=0.6):
    
    samples, log_probs = [], []

    for _ in range(n_samples):
        _, logp, sample = translate(model, text, temperature=temperature)
        samples.append(np.squeeze(sample.numpy()).tolist())
        log_probs.append(logp)
                
    return samples, log_probs

def jaccard_similarity(candidate, reference):

    candidate_set = set(candidate)
    reference_set = set(reference)
    common_tokens = candidate_set.intersection(reference_set)
    all_tokens = candidate_set.union(reference_set)
    overlap = len(common_tokens) / len(all_tokens)
        
    return overlap

def rouge1_similarity(candidate, reference):
    candidate_word_counts = Counter(candidate)
    reference_word_counts = Counter(reference)
    overlap = 0
    for token in candidate_word_counts.keys():
        token_count_candidate = candidate_word_counts[token]
        token_count_reference = reference_word_counts.get(token, 0)    
        overlap += min(token_count_candidate, token_count_reference)
    
    precision = overlap / len(candidate) if len(candidate) > 0 else 0
    recall = overlap / len(reference) if len(reference) > 0 else 0
    
    if precision + recall > 0:
        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score
    
    return 0 

def average_overlap(samples, similarity_fn):

    scores = {}

    for index_candidate, candidate in enumerate(samples):    
        overlap = 0

        for index_sample, sample in enumerate(samples):
            if index_sample == index_candidate:
                continue
                
            sample_overlap = similarity_fn(candidate, sample)
            overlap += sample_overlap

        score = overlap / (len(samples) - 1)
        score = round(score, 3)
        scores[index_candidate] = score
        
    return scores

def weighted_avg_overlap(samples, log_probs, similarity_fn):
    
    scores = {}
    for index_candidate, candidate in enumerate(samples):    
        overlap, weight_sum = 0.0, 0.0
        for index_sample, (sample, logp) in enumerate(zip(samples, log_probs)):          
            if index_candidate == index_sample:
                continue

            sample_p = float(np.exp(logp))
            weight_sum += sample_p
            sample_overlap = similarity_fn(candidate, sample)
            overlap += sample_p * sample_overlap

        score = overlap / weight_sum
        score = round(score, 3)
        scores[index_candidate] = score
    
    return scores

def mbr_decode(model, text, n_samples=5, temperature=0.6, similarity_fn=jaccard_similarity):

    samples, log_probs = generate_samples(model, text, n_samples=n_samples, temperature=temperature)
    scores = weighted_avg_overlap(samples, log_probs, similarity_fn)
    decoded_translations = [tokens_to_text(s, id_to_word).numpy().decode('utf-8') for s in samples]
    max_score_key = max(scores, key=lambda k: scores[k])
    translation = decoded_translations[max_score_key]
    
    return translation, decoded_translations

def main():

    to_translate, sr_translation = next(iter(train_data.take(1)))
    translator = Translator(VOCAB_SIZE, UNITS)
    trained_translator, history = compile_and_train(translator)
    encoder = Encoder(VOCAB_SIZE, UNITS)
    encoder_output = encoder(to_translate)

    attention_layer = CrossAttention(UNITS)
    sr_translation_embed = tf.keras.layers.Embedding(VOCAB_SIZE, output_dim=UNITS, mask_zero=True)(sr_translation)
    attention_result = attention_layer(encoder_output, sr_translation_embed)
    decoder = Decoder(VOCAB_SIZE, UNITS)
    logits = decoder(encoder_output, sr_translation)
    print(logits)

    translator = Translator(VOCAB_SIZE, UNITS)

    # Compute the logits for every word in the vocabulary
    logits = translator((to_translate, sr_translation))
    print(logits)
    trained_translator, history = compile_and_train(translator)
    
    # Experiment with different temperatures (0 indicates greedy)
    temp = 0.7
    english_sentence = "I love languages"
    translation, candidates = mbr_decode(trained_translator, english_sentence, n_samples=10, temperature=0.6)
    print("Translation candidates:")
    for c in candidates:
        print(c)

    print(f"\nSelected translation: {translation}")

    #########


    


main()
