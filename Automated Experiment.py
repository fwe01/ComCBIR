import multiprocessing
import pickle
import os
import time
import tensorflow as tf
import tensorflow.keras.applications as applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
    ReLU,
    InputLayer,
    Input,
)
from ContrastiveLoss import ContrastiveLoss
import numpy as np
import random
from tqdm import tqdm

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
print("GPUs Available: ", tf.config.list_physical_devices("GPU"))

if len(tf.config.list_physical_devices("GPU")) == 0:
    raise Exception("Warning Environtment Without GPU!!!")

MOBILENET = "mobilenet"
MOBILENETV2 = "mobilenetv2"
MOBILENETV3SMALL = "mobilenetv3small"
MOBILENETV3LARGE = "mobilenetv3large"
EFFICIENTNETB0 = "efficientnetb0"
EFFICIENTNETB0V2 = "efficientnetb0v2"

# setting
NUM_FEATURES = 16
# NUM_CLASS = 8 #kvasir
NUM_CLASS = 7 #demamnist
BATCH_SIZE = 256
CLASS_PER_BATCH = BATCH_SIZE // NUM_CLASS
EPOCH = 200
LEARNING_RATE = 0.0001
ACTIVATIONS = "relu"


AT = tf.data.AUTOTUNE
BUFFER = 1000

STEPS_PER_EPOCH = 3600 // BATCH_SIZE  # mau berapa gambar setiap epoch
VALIDATION_STEPS = 400 // BATCH_SIZE

INPUT_SIZE = (224, 224, 3)
SEED_NUMBER = 42


def set_random_seed(seed=SEED_NUMBER):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_class_datasets(dataset):
    class_datasets = []
    for class_num in range(NUM_CLASS):
        class_dataset =[(a, b) for a, b in dataset if (b == class_num)]
        class_datasets.append(class_dataset)

    return class_datasets


def take_single_batch(class_datasets):
    batch_result = []
    for class_num in range(NUM_CLASS):
        class_batch = random.sample(class_datasets[class_num], CLASS_PER_BATCH)

        batch_result.extend(class_batch)

    random.shuffle(batch_result)

    images = tf.stack([a[0] for a in batch_result])
    labels = tf.stack([a[1] for a in batch_result])

    return images, labels


def select_model_and_preprocessor(model_name, num_feature=NUM_FEATURES):
    base_model = None
    set_random_seed()
    if model_name == MOBILENET:
        base_model = applications.MobileNet(
            input_shape=INPUT_SIZE, weights="imagenet", include_top=False
        )
        preprocessor = applications.mobilenet.preprocess_input
    elif model_name == MOBILENETV2:
        base_model = applications.MobileNetV2(
            input_shape=INPUT_SIZE, weights="imagenet", include_top=False
        )
        preprocessor = applications.mobilenet_v2.preprocess_input
    elif model_name == MOBILENETV3SMALL:
        base_model = applications.MobileNetV3Small(
            input_shape=INPUT_SIZE, weights="imagenet", include_top=False
        )
        preprocessor = applications.mobilenet_v2.preprocess_input
    elif model_name == MOBILENETV3LARGE:
        base_model = applications.MobileNetV3Large(
            input_shape=INPUT_SIZE, weights="imagenet", include_top=False
        )
        preprocessor = applications.mobilenet_v2.preprocess_input
    elif model_name == EFFICIENTNETB0:
        base_model = applications.EfficientNetB0(
            input_shape=INPUT_SIZE, weights="imagenet", include_top=False
        )
        preprocessor = applications.efficientnet.preprocess_input
    elif model_name == EFFICIENTNETB0V2:
        base_model = applications.EfficientNetV2B0(
            input_shape=INPUT_SIZE, weights="imagenet", include_top=False
        )
        preprocessor = applications.efficientnet.preprocess_input
    elif base_model == None:
        raise Exception("Model name does not exist")

    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_feature, activation=ACTIVATIONS)(x)
    model = Model(inputs=base_model.input, outputs=x)

    # model.summary()

    return base_model, model, preprocessor


def save_model(model, model_name, dataset, feature_num=NUM_FEATURES, epoch=EPOCH, margin=10):
    model_filename = f"automated_{dataset}_{model_name}_updated_negative_loss_function_{str(margin).replace('.', '_')}_{ACTIVATIONS}_{feature_num}_{str(LEARNING_RATE).replace('.', '_')}_{BATCH_SIZE}_{epoch}.h5"
    model_path = f"../Model/{model_filename}"

    if os.path.exists(model_path):
        print("saved model exists, please change model name")
    else:
        model.save(model_path)
        print(f"model saved to {model_filename}")


# @tf.function
def train_batch(model, loss_class, optimizer, images, y_true):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_class(y_true, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    # Check for NaNs in gradients
    for grad in gradients:
        if grad is not None and tf.reduce_any(tf.math.is_nan(grad)):
            tf.print(f"NaN detected in intra_gradient! Loss : {loss}")
            raise Exception("NaN Grad Detected")

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    del tape

    return loss


# training and fine-tuning
def train_model(model, class_datasets, epoch=EPOCH, margin=10, verbose=False):
    loss_class = ContrastiveLoss(margin=margin, verbose=False)
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

    overall_progress = []
    for current_epoch in range(epoch):
        print(f"Epoch {current_epoch + 1}/{epoch}")
        print(50 * "=")
        epoch_progress = []

        for step in range(STEPS_PER_EPOCH):
            images, y_true = take_single_batch(class_datasets)
            # save weight before nan

            loss = train_batch(model, loss_class, optimizer, images, y_true)

            if verbose:
                print(f"Step {step}, Loss : {loss}")

            epoch_progress.append([current_epoch, step, loss])

        temp = np.array(epoch_progress)

        print(f"Avg Loss : {sum(temp[:, 2] / len(epoch_progress))}")


# Define a simple function that each process will run
def worker(identifier, model_name, num_feature, margin, dataset):
    print(
        f"Worker {identifier} starting. Training model {model_name} with {num_feature} feature using {dataset} and margin {margin}"
    )

    dataset_file = open(f'../Dataset/{dataset}/training', 'rb')
    train_dataset = pickle.load(dataset_file)
    dataset_file.close()
    class_datasets = create_class_datasets(train_dataset)

    base_model, model, preprocessor = select_model_and_preprocessor(
        model_name, num_feature
    )

    train_model(model, class_datasets, epoch=100, margin=margin)
    save_model(model, model_name, dataset, num_feature, epoch=100, margin=margin)

    train_model(model, class_datasets, epoch=100, margin=margin)
    save_model(model, model_name, dataset, num_feature, epoch=200, margin=margin)

    print(f"Worker {identifier} finished.")


if __name__ == "__main__":
    # Create a list of processes

    datasets = [
        'kvasir_scaled_min1_to_1_augmented',
        'dermamnist_scaled_min1_to_1_augmented',
    ]
    
    models = [
        MOBILENET,
        MOBILENETV2,
        MOBILENETV3LARGE,
        EFFICIENTNETB0,
        EFFICIENTNETB0V2,
    ]
    num_features = [8, 16, 24, 32, 64]

    margins = [1] #not used

    # Initialize processes
    curr_process_id = 1
    for dataset in datasets :
        for model in models:
            for num_feature in num_features:
                for margin in margins:
                    process = multiprocessing.Process(
                        target=worker, args=(curr_process_id, model, num_feature, margin, dataset)
                    )
                    process.start()
                    process.join()
                    curr_process_id += 1

    print("All workers are done.")
