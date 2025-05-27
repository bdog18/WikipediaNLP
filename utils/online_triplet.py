import os
import numpy as np
import tensorflow as tf
import json
from keras import layers, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
# Turn off warnings for Tensorflow 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info & warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging
tf_logger = logging.getLogger("tensorflow")
tf_logger.setLevel(logging.ERROR)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout1 = layers.Dropout(rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(rate),
            layers.Dense(embed_dim),
        ])
        self.dropout2 = layers.Dropout(rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        out1 = self.norm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1, training=training)
        return self.norm2(out1 + self.dropout2(ffn_output, training=training))


class CustomEncoder(Model):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers=2):
        super().__init__()
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.pooling = layers.GlobalAveragePooling1D()

    def call(self, x, training=False):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        x = self.pooling(x)
        return tf.math.l2_normalize(x, axis=1)


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
    

class TripletAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, encoder):
        super().__init__()
        self.val_dataset = val_dataset
        self.encoder = encoder

    def on_epoch_end(self, epoch, logs=None):
        correct = 0
        total = 0
        for a, p, n in self.val_dataset:
            a_embed = self.encoder(a, training=False)
            p_embed = self.encoder(p, training=False)
            n_embed = self.encoder(n, training=False)

            pos_dist = tf.reduce_sum(tf.square(a_embed - p_embed), axis=1)
            neg_dist = tf.reduce_sum(tf.square(a_embed - n_embed), axis=1)

            correct += tf.reduce_sum(tf.cast(pos_dist < neg_dist, tf.float32)).numpy()
            total += a.shape[0]
        acc = correct / total
        print(f"Triplet Accuracy (val) @ Epoch {epoch+1}: {acc:.4f}")


def load_triplet_validation_dataset(json_dir, vectorizer, batch_size=512):
    # Collect and cache the list of files once
    file_paths = [
        os.path.join(json_dir, f)
        for f in sorted(os.listdir(json_dir))
        if f.endswith(".json")
    ]

    # Generator uses the cached file list
    def generator():
        for file in file_paths:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    t = json.loads(line)
                    yield t["anchor"], t["positive"], t["negative"]

    def vectorize_fn(a, p, n): 
        return vectorizer(a), vectorizer(p), vectorizer(n)
    
    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.map(vectorize_fn).batch(batch_size).prefetch(tf.data.AUTOTUNE)



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


if __name__ == '__main__':
    input_dir = "../data/temp/anchor_positive_pairs"
    val_dir = "../data/temp/val_anchor_positive_pairs"
    weights_dir = "../data/custom_model/encoder_weights"
    vectorizer_path = "../data/custom_model/saved_vectorizer"
    vocab_size = 75000
    max_len = 256
    embed_dim = 512
    num_heads = 8
    ff_dim = 2 * embed_dim
    batch_size = 512
    num_epochs = 20
    total_lines = 4091164
    total_lines = 4254476

    # def count_total_lines(directory, file_ext=".json"):
    #     total_lines = 0
    #     for file in os.listdir(directory):
    #         if file.endswith(file_ext):
    #             file_path = os.path.join(directory, file)
    #             with open(file_path, "r", encoding="utf-8") as f:
    #                 total_lines += sum(1 for _ in f)
    #     return total_lines
    
    # total_lines = count_total_lines(input_dir)
    print(f"Total lines in '{input_dir}': {total_lines}")

    # Load or create vectorizer
    if os.path.exists(vectorizer_path):
        vectorizer = tf.keras.models.load_model(vectorizer_path)
    else:
        create_vectorizer(input_dir, vectorizer_path, vocab_size, max_len)
        vectorizer = tf.keras.models.load_model(vectorizer_path)

    # Prepare training dataset
    dataset = load_ap_dataset(input_dir, vectorizer, batch_size)

    # Prepare validation dataset
    val_dataset = load_triplet_validation_dataset(val_dir, vectorizer, batch_size=512)

    # Initialize encoder and trainer
    encoder = CustomEncoder(vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers=2)
    
    # if os.path.exists("../data/custom_model/encoder_weights/best_encoder.weights.h5"):
    #     encoder.load_weights("../data/custom_model/encoder_weights/best_encoder.weights.h5")
    #     print("Loaded existing best weights.")

    trainer = OnlineTripletTrainer(encoder)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0))

    # Define callbacks for early stopping and checkpointing
    callbacks = [
        EarlyStopping(monitor="loss", patience=3),
        ModelCheckpoint(f"{weights_dir}/best_encoder.weights.h5", 
                        monitor="loss", 
                        save_best_only=True,
                        save_weights_only=True),
        TripletAccuracyCallback(val_dataset, encoder)
    ]

    # Begin training
    trainer.fit(dataset.repeat(), 
                epochs=num_epochs, 
                steps_per_epoch=total_lines // batch_size, 
                callbacks=callbacks)

    # Save final model weights
    encoder.save_weights(f"{weights_dir}/final_encoder.weights.h5")
    print("Online triplet mining training complete.")
