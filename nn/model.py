import json
import random
import shutil
from pathlib import Path

from argser import SubCommands
from keras import Sequential
from keras.engine.saving import save_model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img

from nn.consts import CLASSES, IMG_ROOT, IM_HEIGHT, IM_WIDTH, LABELS_FILE, MODEL_PATH


sub = SubCommands()


def make_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(IM_HEIGHT, IM_WIDTH, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(3, activation="sigmoid"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def preview(generator):
    img_path = next(IMG_ROOT.glob("*.png"))
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    save_dir = Path("data/preview")
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir()
    flow = generator.flow(x, batch_size=1, save_to_dir=str(save_dir), save_prefix="test", save_format="png")
    for i, _ in enumerate(flow):
        if i > 20:
            break


@sub.add
def train(batch_size=32, load_weights=True, data_dir="data/train", val_dir="data/validation", epochs=1):
    model = make_model()
    model.summary()
    if load_weights:
        model.load_weights(MODEL_PATH)

    img_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1.0 / 255,
        shear_range=0.02,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="constant",
        cval=255,
    )
    train_gen = img_gen.flow_from_directory(
        directory=data_dir, target_size=(IM_HEIGHT, IM_WIDTH), batch_size=batch_size, classes=CLASSES
    )
    val_gen = img_gen.flow_from_directory(
        directory=val_dir, target_size=(IM_HEIGHT, IM_WIDTH), batch_size=batch_size, classes=CLASSES
    )
    model.fit_generator(train_gen, validation_data=val_gen, epochs=epochs)
    save_model(model, MODEL_PATH)


@sub.add
def datagen(target_dir="data"):
    target_dir = Path(target_dir)
    dataset_types = ("train", "validation", "test")
    dataset_types_pick = [0] * 9 + [1] * 2 + [2] * 1

    # recreate
    for klass in CLASSES:
        for dt in dataset_types:
            dir = target_dir / dt / klass
            if dir.exists():
                shutil.rmtree(dir)
            dir.mkdir(exist_ok=True, parents=True)

    labels: dict = json.loads(LABELS_FILE.read_text())
    for path, index in labels.items():
        klass = CLASSES[index]
        dt_inx = random.choice(dataset_types_pick)
        dt = dataset_types[dt_inx]
        shutil.copy(path, target_dir / dt / klass)


if __name__ == "__main__":
    sub.parse()
