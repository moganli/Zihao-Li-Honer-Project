import os
import numpy as np
import tensorflow as tf
#from tensorflow import keras
from PIL import Image
# toimage
# 更改为
# Image.fromarray
import glob
from gan import Generator, Discriminator
from data import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocessed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocessed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocessed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)


def celoss_one(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zero(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def d_loss_fn(G, D, batch_z, batch_x, training):
    gimage = G(batch_z, training)
    d_fake_logits = D(gimage, training)
    d_real_logits = D(batch_x, training)

    d_loss_real = celoss_one(d_real_logits)
    d_loss_fake = celoss_zero(d_fake_logits)

    loss = d_loss_real + d_loss_fake
    return loss


def g_loss_fn(G, D, batch_z, training):
    fake_image = G(batch_z, training)
    d_fake_logits = D(fake_image, training)
    loss = celoss_one(d_fake_logits)
    return loss


def main():
    # 初始随机值
    tf.random.set_seed(1216)
    np.random.seed(1216)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    # 重要训练残数
    z_dim = 120  # 隐藏向量z的长度
    epochs = 3000000  # 训练步数
    batch_size = 64  # batch size
    learning_rate = 0.0002
    is_training = True


    # 导入图片数据并处理
    img_path = glob.glob(r'E:\dataSet\tiny-imagenet-200\train\n01443537\images\*.JPEG')
    dataset, img_shape = make_anime_dataset(img_path, batch_size)
    print(dataset, img_shape)
    sample = next(iter(dataset))
    db_iter = iter(dataset.repeat())
    #print(sample.shape,tf.reduce_max(sample),tf.reduce_min(sample))

    G = Generator()
    G.build(input_shape=(None, z_dim))

    D = Discriminator()
    D.build(input_shape=(None, 64,64,3))

    # 优化器
    Goptimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    Doptimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)
        batch_x = next(db_iter)

        # 训练辨别器
        with tf.GradientTape() as tape:  # 梯度带， 求导
            d_loss = d_loss_fn(G, D, batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, D.trainable_variables)
        # 迭代更新
        Doptimizer.apply_gradients(zip(grads, D.trainable_variables))

        # 训练生成器
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(G, D, batch_z, is_training)
        grads = tape.gradient(g_loss, G.trainable_variables)
        Goptimizer.apply_gradients(zip(grads, G.trainable_variables))
        #每100次 存储一次用来观察
        if epoch % 100 == 0:
            print(epoch,'D loss',d_loss,'G loss', g_loss)

            z = tf.random.normal([120, z_dim])
            fake_image = G(z, training=False)
            img_path = os.path.join(r'C:\Users\86135\Desktop\Zihao-Li-Honer-Project\images', 'gan_%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')


if __name__ == '__main__':
    main()
