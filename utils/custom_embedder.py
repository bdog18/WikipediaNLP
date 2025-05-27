import os
import numpy as np
import tensorflow as tf
import json
from keras import layers, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))


class CustomEncoder(Model):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers=2):
        super().__init__()
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers+1)]
        self.pooling = layers.GlobalAveragePooling1D()

    def call(self, x, training=False):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return self.pooling(x)


def triplet_loss(anchor, positive, negative, margin=0.3):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)


class TripletTrainer(Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        anchor, positive, negative = data
        with tf.GradientTape() as tape:
            a_embed = self.encoder(anchor, training=True)
            p_embed = self.encoder(positive, training=True)
            n_embed = self.encoder(negative, training=True)
            loss = triplet_loss(a_embed, p_embed, n_embed)

        grads = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def create_vectorizer(input_dir, vectorizer_path, vocab_size, max_len):
    print("Fitting vectorizer on all data (streamed)")
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len
    )

    def text_generator(input_dir):
        files = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])
        for part in files:
            with open(os.path.join(input_dir, part), "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    yield item["anchor"]
                    yield item["positive"]
                    yield item["negative"]

    text_ds = tf.data.Dataset.from_generator(
        lambda: text_generator(input_dir),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    ).batch(512)

    vectorizer.adapt(text_ds)

    print("Saving vectorizer")
    model_for_saving = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer
    ])
    model_for_saving.save(vectorizer_path)

    return


def load_triplet_dataset(json_dir, vectorizer, batch_size):
    anchors, positives, negatives = [], [], []
    for file in sorted(os.listdir(json_dir)):
        if not file.endswith(".json"): continue
        with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                anchors.append(t["anchor"])
                positives.append(t["positive"])
                negatives.append(t["negative"])

    anchors_seq = vectorizer(tf.constant(anchors))
    positives_seq = vectorizer(tf.constant(positives))
    negatives_seq = vectorizer(tf.constant(negatives))

    dataset = tf.data.Dataset.from_tensor_slices((anchors_seq, positives_seq, negatives_seq))
    dataset = dataset.shuffle(10000).batch(batch_size)
    return dataset

def load_triplet_dataset_streamed(json_dir, vectorizer, batch_size):
    def generator():
        for file in sorted(os.listdir(json_dir)):
            if not file.endswith(".json"): continue
            with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    t = json.loads(line)
                    yield t["anchor"], t["positive"], t["negative"]

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    # Vectorize in-place
    def vectorize_fn(anchor, pos, neg):
        return vectorizer(anchor), vectorizer(pos), vectorizer(neg)

    return dataset.map(vectorize_fn, num_parallel_calls=tf.data.AUTOTUNE) \
                  .shuffle(10000) \
                  .batch(batch_size) \
                  .prefetch(tf.data.AUTOTUNE)




def train_with_config(config, vectorizer, input_dir, batch_size, steps_per_epoch, run_id):
    print(f"🔧 Training config {run_id}: {config}")

    encoder = CustomEncoder(
        vocab_size=30000,
        max_len=32,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"]
    )

    model = TripletTrainer(encoder)
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]))

    # Load streaming dataset
    train_dataset = load_triplet_dataset_streamed(input_dir, vectorizer, batch_size)

    # Callbacks
    checkpoint_path = f"../data/custom_model/gridsearch/encoder_{run_id}.weights.h5"
    callbacks = [
        EarlyStopping(monitor="loss", patience=2),
        ModelCheckpoint(checkpoint_path, monitor="loss", save_best_only=True, save_weights_only=True)
    ]

    # Train
    history = model.fit(
        train_dataset.repeat(),
        steps_per_epoch=steps_per_epoch,
        epochs=5,
        callbacks=callbacks
    )

    best_loss = min(history.history["loss"])
    return best_loss, config, checkpoint_path


# -----------------------------
# Main Script
# -----------------------------
if __name__ == '__main__':
    # Parameters
    input_dir = "../data/processed/triplets/parts"
    vectorizer_path = "../data/custom_model/saved_vectorizer"
    vocab_size = 30000
    max_len = 32
    embed_dim = 128
    num_heads = 4
    ff_dim = 256
    batch_size = 64
    num_epochs = 10
    weights_path = "../data/custom_model/encoder_weights.h5"

    # Load or create vectorizer
    if os.path.exists(vectorizer_path):
        print("Loading saved vectorizer")
        vectorizer = tf.keras.models.load_model(vectorizer_path)
    else:
        vectorizer = create_vectorizer(input_dir, vectorizer_path, vocab_size, max_len)


    # Load dataset
    train_dataset = load_triplet_dataset(input_dir, vectorizer, batch_size)

    # Model setup
    encoder = CustomEncoder(vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers=2)
    trainer = TripletTrainer(encoder)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="loss", patience=2),
        ModelCheckpoint("../data/custom_model/best_encoder.keras", monitor="loss", save_best_only=True)
    ]

    # Training
    trainer.fit(train_dataset, epochs=num_epochs, callbacks=callbacks)

    # Save final encoder weights
    encoder.save_weights(weights_path)
    print("Training complete and weights saved.")