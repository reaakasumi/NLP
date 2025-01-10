import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import spacy

df_train = pd.read_csv("data/train_preprocessed.csv")
df_test = pd.read_csv("data/test_preprocessed.csv")

# function to compute overlap of lemmas
def compute_overlap(reference_tokens, hypothesis_tokens):
    ref_set = set(reference_tokens)
    hyp_set = set(hypothesis_tokens)
    overlap = ref_set.intersection(hyp_set)
    return len(overlap) / len(hyp_set) if hyp_set else 0


# function to calculate similarity after semantic embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(reference, hypothesis):
    ref_embedding = model.encode([reference])
    hyp_embedding = model.encode([hypothesis])
    similarity = cosine_similarity(ref_embedding, hyp_embedding)
    return similarity[0][0]

# function to calculate bleu score
def bleu_score(reference, hypothesis):
    return sentence_bleu(reference, hypothesis, smoothing_function = SmoothingFunction().method4)


# function to calculate overlap of named entities
nlp = spacy.load("en_core_web_sm")

def entity_overlap(reference, hypothesis):

    ref = nlp(reference)
    entities_ref = set(ent.text for ent in ref.ents)

    hyp = nlp(hypothesis)
    entities_hyp = set(ent.text for ent in hyp.ents)

    overlap = entities_hyp.intersection(entities_ref)
    return len(overlap) / len(entities_hyp) if entities_hyp else 0


# Extract features for training set
df_train['overlap'] = df_train.apply(lambda row: compute_overlap(row['tgt_src_lemmas'], row['hyp_lemmas']), axis=1)
df_train['semantic_similarity'] = df_train.apply(lambda row: semantic_similarity(row['tgt_src'], row['hyp']), axis=1)
df_train['bleu_score'] = df_train.apply(lambda row: bleu_score(row['tgt_src_lemmas'], row['hyp_lemmas']), axis=1)
df_train['entity_overlap'] = df_train.apply(lambda row: entity_overlap(row['tgt_src'], row['hyp']), axis=1)

# Extract features for test set
df_test['overlap'] = df_test.apply(lambda row: compute_overlap(row['tgt_src_lemmas'], row['hyp_lemmas']), axis=1)
df_test['semantic_similarity'] = df_test.apply(lambda row: semantic_similarity(row['tgt_src'], row['hyp']), axis=1)
df_test['bleu_score'] = df_test.apply(lambda row: bleu_score(row['tgt_src_lemmas'], row['hyp_lemmas']), axis=1)
df_test['entity_overlap'] = df_test.apply(lambda row: entity_overlap(row['tgt_src'], row['hyp']), axis=1)


# store dataset with new features
df_train.to_csv("data/train_features.csv")
df_test.to_csv("data/test_features.csv")
