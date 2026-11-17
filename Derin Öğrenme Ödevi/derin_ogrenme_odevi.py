import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Grafikler ciktilar klasörüne de kaydediliyor
os.makedirs("ciktilar", exist_ok=True)



# 3. Bölüm: Konvolüsyon işlemi

print("\n----- 3. BÖLÜM: KONVOLÜSYON -----")

tf.keras.utils.set_random_seed(42)

conv = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer='he_normal'
)

dummy_input = tf.random.normal((1, 128, 128, 1))
output = conv(dummy_input)

weights, biases = conv.get_weights()

print("Giriş boyutu:", dummy_input.shape)
print("Çıkış boyutu:", output.shape)
print("Filtre ağırlıklarının boyutu:", weights.shape)
print("Bias boyutu:", biases.shape)
print("İlk filtre:")
print(weights[:, :, 0, 0])

# 16 filtrenin başlangıç ağırlıklarını görselleştirme
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(weights[:, :, 0, i], cmap='gray')
    plt.title("Filtre " + str(i + 1))
    plt.axis('off')
plt.suptitle("16 filtrenin başlangıç ağırlıkları")
plt.tight_layout()
plt.savefig("ciktilar/filtreler.png")
plt.show()



# 4. Bölüm: Max Pooling

print("\n----- 4. BÖLÜM: MAX POOLING -----")

input_data = tf.constant([
    [[1., 3., 2., 4.],
     [5., 6., 1., 2.],
     [7., 2., 8., 3.],
     [4., 1., 5., 9.]]
])
input_data = tf.reshape(input_data, (1, 4, 4, 1))

pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
output = pool(input_data)

print("Giriş:")
print(input_data.numpy()[0, :, :, 0])
print("Pooling sonucu:")
print(output.numpy()[0, :, :, 0])
print("Giriş boyutu:", input_data.shape, " Çıkış boyutu:", output.shape)



# 5. Bölüm: ReLU ve Softmax

print("\n----- 5. BÖLÜM: RELU -----")

x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
relu = tf.keras.layers.ReLU()
y = relu(x)
print("Giriş:", x.numpy())
print("ReLU çıkışı:", y.numpy())

print("\n----- 5. BÖLÜM: SOFTMAX -----")

logits = tf.constant([[2.0, 1.0, 0.1]])
softmax = tf.keras.layers.Softmax()
probabilities = softmax(logits)
print("Softmax sonuçları:")
print(probabilities.numpy())
print("Olasılıkların toplamı:")
print(tf.reduce_sum(probabilities).numpy())



# 6. Bölüm: Fashion-MNIST ile CNN modeli

print("\n----- 6. BÖLÜM: CNN MODELİ -----")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalizasyon
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Kanal boyutu ekleme
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("Eğitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)


# Deneyde de aynı modeli kullanacağım için modeli fonksiyon olarak yazdım.
# ilk_filtre: ilk konvolüsyon katmanındaki filtre sayısı
def model_olustur(ilk_filtre):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(ilk_filtre, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def egit(ilk_filtre):
    tf.keras.utils.set_random_seed(42)   # iki deneyde de aynı başlangıç
    model = model_olustur(ilk_filtre)
    model.summary()

    baslangic = time.time()
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    sure = time.time() - baslangic

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    return history, sure, test_accuracy


# Deney 1: ilk konvolüsyonda 8 filtre (ödevdeki asıl model)
history8, sure8, test8 = egit(8)

# Eğitim ve doğrulama grafikleri
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history8.history['accuracy'], label='Training')
plt.plot(history8.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history8.history['loss'], label='Training')
plt.plot(history8.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("ciktilar/egitim_grafikleri.png")
plt.show()



# 8. Bölüm: Deney (8 ve 16 filtre)

print("\n----- 8. BÖLÜM: DENEY 2 (16 FİLTRE) -----")

history16, sure16, test16 = egit(16)

print("\n----- DENEY SONUÇLARI -----")
print("Ölçüt               Deney 1 (8)   Deney 2 (16)")
print("Eğitim doğruluğu    %.4f        %.4f" % (history8.history['accuracy'][-1], history16.history['accuracy'][-1]))
print("Test doğruluğu      %.4f        %.4f" % (test8, test16))
print("Eğitim süresi (sn)  %.1f          %.1f" % (sure8, sure16))
