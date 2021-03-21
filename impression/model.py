import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def export_model(path, train_path='train', test_path='test',
                 validation_split=0.2, seed=42,
                 batch_size=32,
                 max_features=5000, embedding_dim=128, sequence_length=500) :

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_path, batch_size=batch_size, validation_split=validation_split, subset='training', seed=seed)

    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_path, batch_size=batch_size, validation_split=validation_split, subset='validation', seed=seed)

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(test_path, batch_size=batch_size)

    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(4)
    ])

    model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    model.evaluate(test_ds)

    model_export = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('softmax')
    ])
    model_export.save(path)