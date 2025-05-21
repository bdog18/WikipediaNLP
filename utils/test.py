import os
import numpy as np
import tensorflow as tf
import json
from keras import layers, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# -----------------------------
# Transformer Encoder Components
# -----------------------------
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        # Multi-head self-attention layer
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        # Layer normalization and dropout for regularization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # Apply self-attention
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        # Apply FFN and second normalization
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))


class CustomEncoder(Model):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers=2):
        super().__init__()
        # Token and positional embeddings
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        # Stack multiple Transformer blocks
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        # Use global average pooling to get sentence embedding
        self.pooling = layers.GlobalAveragePooling1D()

    def call(self, x, training=False):
        # Create position indices and apply embeddings
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        # Pass through transformer layers
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return self.pooling(x)


# -----------------------------
# Online Triplet Mining Trainer
# -----------------------------
class OnlineTripletTrainer(Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        anchor, positive = data  # negatives are mined dynamically

        with tf.GradientTape() as tape:
            # Encode anchor and positive inputs
            a_embed = self.encoder(anchor, training=True)
            p_embed = self.encoder(positive, training=True)

            # Normalize embeddings for cosine similarity
            a_norm = tf.math.l2_normalize(a_embed, axis=1)
            p_norm = tf.math.l2_normalize(p_embed, axis=1)

            # Compute cosine similarity matrix between anchors and positives
            sim_matrix = tf.matmul(a_norm, tf.transpose(p_norm))

            batch_size = tf.shape(a_embed)[0]
            pos_indices = tf.range(batch_size)
            # Extract diagonal for true positive similarities
            pos_scores = tf.gather_nd(sim_matrix, tf.stack([pos_indices, pos_indices], axis=1))

            # Mask out self-pairs to avoid selecting them as negatives
            neg_mask = tf.eye(batch_size, dtype=tf.bool)
            sim_matrix = tf.where(neg_mask, tf.constant(-1e9, dtype=sim_matrix.dtype), sim_matrix)

            # Select hardest negative for each anchor (highest similarity)
            neg_indices = tf.argmax(sim_matrix, axis=1)
            neg_indices = tf.cast(neg_indices, dtype=tf.int32)
            neg_scores = tf.gather_nd(sim_matrix, tf.stack([pos_indices, neg_indices], axis=1))

            # Compute triplet loss: max(pos - neg + margin, 0)
            margin = 0.3
            losses = tf.maximum(pos_scores - neg_scores + margin, 0.0)
            loss = tf.reduce_mean(losses)
            loss = tf.cast(loss, tf.float32)

        # Apply gradients to encoder
        grads = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# -----------------------------
# Vectorizer Setup
# -----------------------------
def create_vectorizer(input_dir, vectorizer_path, vocab_size, max_len):
    print("Fitting vectorizer on anchor-positive pairs")
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len
    )

    # Generator for anchor and positive text
    def text_generator():
        for file in sorted(os.listdir(input_dir)):
            if not file.endswith(".json"): continue
            with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    t = json.loads(line)
                    yield t["anchor"]
                    yield t["positive"]

    text_ds = tf.data.Dataset.from_generator(
        text_generator,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    ).batch(512)

    # Fit vectorizer and save for later reuse
    vectorizer.adapt(text_ds)
    model_for_saving = tf.keras.Sequential([tf.keras.Input(shape=(1,), dtype=tf.string), vectorizer])
    model_for_saving.save(vectorizer_path)
    return


# -----------------------------
# Dataset Loader for Anchor-Positive Pairs
# -----------------------------
def load_ap_dataset(json_dir, vectorizer, batch_size):
    # Generator that yields anchor-positive pairs
    def generator():
        for file in sorted(os.listdir(json_dir)):
            if not file.endswith(".json"): continue
            with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    t = json.loads(line)
                    yield t["anchor"], t["positive"]

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string)
    )

    # Apply vectorization, shuffle, batch, and prefetch
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    def vectorize_fn(a, p): return vectorizer(a), vectorizer(p)
    return dataset.map(vectorize_fn).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# -----------------------------
# Training Script
# -----------------------------
if __name__ == '__main__':
    input_dir = "../data/processed/triplets/parts"
    weights_dir = "../data/custom_model/encoder_weights"
    vectorizer_path = "../data/custom_model/saved_vectorizer"
    vocab_size = 50000
    max_len = 64
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    batch_size = 384
    num_epochs = 20
    total_lines = 4091164

    # Load or create vectorizer
    if os.path.exists(vectorizer_path):
        vectorizer = tf.keras.models.load_model(vectorizer_path)
    else:
        create_vectorizer(input_dir, vectorizer_path, vocab_size, max_len)
        vectorizer = tf.keras.models.load_model(vectorizer_path)

    # Prepare training dataset
    dataset = load_ap_dataset(input_dir, vectorizer, batch_size)

    # Initialize encoder and trainer
    encoder = CustomEncoder(vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers=2)
    trainer = OnlineTripletTrainer(encoder)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    # Define callbacks for early stopping and checkpointing
    callbacks = [
        EarlyStopping(monitor="loss", patience=2),
        ModelCheckpoint(f"{weights_dir}/best_encoder.weights.h5", 
                        monitor="loss", 
                        save_best_only=True,
                        save_weights_only=True)
    ]

    # if os.path.exists("../data/custom_model/encoder_weights/best_encoder.weights.h5"):
    #     encoder.load_weights("../data/custom_model/encoder_weights/best_encoder.weights.h5")
    #     print("Loaded existing best weights.")

    # Begin training
    trainer.fit(dataset.repeat(), 
                epochs=num_epochs, 
                steps_per_epoch=total_lines // batch_size, 
                callbacks=callbacks)

    # Save final model weights
    encoder.save_weights(f"{weights_dir}/final_encoder.weights.h5")
    print("Online triplet mining training complete.")
