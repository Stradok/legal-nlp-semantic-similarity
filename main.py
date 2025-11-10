# ==============================
#   LEGAL CLAUSE SIMILARITY
#   BASELINES: BiLSTM + Attention Encoder
#   GPT-5 (ChatGPT) — Ready to Run Notebook
# ==============================

import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalAveragePooling1D, Attention, Concatenate
from tensorflow.keras.optimizers import Adam

# =========================================
# STEP 1 — LOAD ALL CSVs INTO A SINGLE DATAFRAME
# =========================================

folder = "archive"   # CHANGE THIS IF YOUR FOLDER NAME IS DIFFERENT

all_data = []
for file in os.listdir(folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder, file))
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
data = data.dropna()
data.columns = ["clause", "type"]

print("Total clauses:", len(data))
print(data.head())

# =========================================
# STEP 2 — CREATE CLAUSE PAIRS (SIMILAR / NOT SIMILAR)
# =========================================

def create_pairs(df, max_pairs_per_class=400):
    pairs = []
    grouped = df.groupby("type")["clause"].apply(list)

    # Create similar pairs
    for cls, clauses in grouped.items():
        if len(clauses) > 1:
            for i in range(min(len(clauses), max_pairs_per_class) - 1):
                pairs.append((clauses[i], clauses[i+1], 1))

    # Create non-similar pairs
    class_list = list(grouped.items())
    for i in range(max_pairs_per_class * len(class_list)):
        (cls1, c_list1), (cls2, c_list2) = random.sample(class_list, 2)
        c1 = random.choice(c_list1)
        c2 = random.choice(c_list2)
        pairs.append((c1, c2, 0))

    random.shuffle(pairs)
    return pd.DataFrame(pairs, columns=["c1", "c2", "label"])


pairs = create_pairs(data)
print("Total pairs:", len(pairs))
print(pairs.head())

# =========================================
# STEP 3 — TEXT TOKENIZATION
# =========================================

tokenizer = Tokenizer(num_words=20000, oov_token="<UNK>")
tokenizer.fit_on_texts(list(pairs["c1"]) + list(pairs["c2"]))

max_len = 40

X1 = pad_sequences(tokenizer.texts_to_sequences(pairs["c1"]), maxlen=max_len)
X2 = pad_sequences(tokenizer.texts_to_sequences(pairs["c2"]), maxlen=max_len)
y = np.array(pairs["label"])

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2)

# =========================================
# STEP 4 — MODEL 1: BiLSTM Siamese Network
# =========================================

def build_bilstm_model():
    inp1 = Input(shape=(max_len,))
    inp2 = Input(shape=(max_len,))

    embed = Embedding(input_dim=20000, output_dim=128)

    shared_lstm = Bidirectional(LSTM(64, return_sequences=False))

    x1 = shared_lstm(embed(inp1))
    x2 = shared_lstm(embed(inp2))

    merged = Concatenate()([x1, x2])
    out = Dense(64, activation='relu')(merged)
    out = Dropout(0.3)(out)
    out = Dense(1, activation='sigmoid')(out)

    model = Model([inp1, inp2], out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy'])
    return model

bilstm_model = build_bilstm_model()
print(bilstm_model.summary())

history1 = bilstm_model.fit([X1_train, X2_train], y_train, validation_split=0.1, epochs=5, batch_size=32)

# =========================================
# STEP 5 — MODEL 2: Attention Encoder
# =========================================

def build_attention_model():
    inp1 = Input(shape=(max_len,))
    inp2 = Input(shape=(max_len,))

    embed = Embedding(input_dim=20000, output_dim=128)

    enc1 = embed(inp1)
    enc2 = embed(inp2)

    attn = Attention()([enc1, enc2])
    pooled = GlobalAveragePooling1D()(attn)

    x = Dense(128, activation='relu')(pooled)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model([inp1, inp2], out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy'])
    return model

attn_model = build_attention_model()
print(attn_model.summary())

history2 = attn_model.fit([X1_train, X2_train], y_train, validation_split=0.1, epochs=5, batch_size=32)

# =========================================
# STEP 6 — EVALUATION
# =========================================

def evaluate(model, name):
    preds = (model.predict([X1_test, X2_test]) > 0.5).astype(int)
    print("\n==== Results:", name, "====")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))

evaluate(bilstm_model, "BiLSTM")
evaluate(attn_model, "Attention Encoder")
