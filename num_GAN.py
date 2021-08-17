import random
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)


def build_generator():
    noise_shape = (100,)
    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()
    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)


def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='softmax'))
    model.summary()
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)


def build_classifier():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)


def train(epochs, batch_size=128, save_interval=50, keep_from=0):
    optimizer = Adam(0.0002, 0.5)

    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    classifier = build_classifier()
    classifier.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    z = Input(shape=(100,))
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    (X_train, Y_train), (X_train2, Y_train2) = mnist.load_data()
    X_train = np.concatenate((X_train, X_train2))
    Y_train = np.concatenate((Y_train, Y_train2))
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    half_batch = int(batch_size / 2)

    for epoch in range(keep_from, epochs):

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]
        lables = Y_train[idx]
        noise = np.random.normal(0, 1, (half_batch, 100))

        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        c_loss = np.array(classifier.train_on_batch(imgs, lables))

        noise = np.random.normal(0, 1, (batch_size, 100))


        valid_y = np.array([1] * batch_size)

        g_loss = combined.train_on_batch(noise, valid_y)

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] " % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_imgs(epoch)
            generator.save('models/generator_model.h5')
            discriminator.save('models/discriminator_model.h5')
            classifier.save('models/classifier_model.h5')


def save_imgs(epoch):
    generator = load_model('models/generator_model.h5')
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    for i in range(25):
        noise[i, 99] = random.randint(0, 9) / 10
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("output_images/mnist_%d.png" % epoch)
    plt.close()


def generate_numbers ():
    generator_ = load_model('models/generator_model.h5')
    classifier_ = load_model('models/classifier_model.h5')
    vector = randn(10000)
    vector = vector.reshape(100, 100)
    X = generator_.predict(vector)
    preds = classifier_.predict(X)
    num_array = np.zeros((10,28,28,1))
    for i in range(20):
        num_array[np.argmax(preds[i])] = X[i]
    return num_array



train(epochs=100000, batch_size=32, save_interval=500)
