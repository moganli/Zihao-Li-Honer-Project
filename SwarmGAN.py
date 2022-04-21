import glob
import os
from utils import *
from dataloader import *
from gan import Generator, Discriminator

from adabelief_tf import AdaBeliefOptimizer
import matplotlib
import matplotlib.pyplot as plt

class SwarmGAN(object):
    def __init__(self,
                 model_name="SwarmGan",
                 datasetPATH='',
                 outputPATH='',
                 z_dim=100,
                 batch_size=256,
                 G1_learning_rate=0.002,
                 G2_learning_rate=0.0002,
                 G3_learning_rate=0.006,
                 D1_learning_rate=0.002,
                 D2_learning_rate=0.0002,
                 D3_learning_rate=0.006,
                 epochs=25000,
                 disp_freq=100,
                 random_seed=1216,
                 alpha=0.8,
                 beta=0.8,
                 gamma=0.8
                 ):
        self.model_name = model_name
        self.datasetPATH = datasetPATH
        self.outputPATH = outputPATH
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.G1_learning_rate = G1_learning_rate
        self.G2_learning_rate = G2_learning_rate
        self.G3_learning_rate = G3_learning_rate
        self.D1_learning_rate = D1_learning_rate
        self.D2_learning_rate = D2_learning_rate
        self.D3_learning_rate = D3_learning_rate
        self.epochs = epochs
        self.disp_freq = disp_freq
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self):
        is_training = True

        print(
            '   model_name: ', self.model_name,
            '  \ndatasetPATH: ', self.datasetPATH,
            '  \noutputPATH: ', self.outputPATH,
            '   \nG1_learning_rate: ', self.G1_learning_rate,
            '   G2_learning_rate: ', self.G2_learning_rate,
            '   G3_learning_rate: ', self.G3_learning_rate,
            '   \nD1_learning_rate: ', self.D1_learning_rate,
            '   D2_learning_rate: ', self.D2_learning_rate,
            '   D3_learning_rate: ', self.D3_learning_rate,
            '   \nepochs: ', self.epochs,
            '   disp_freq: ', self.disp_freq,
            '   random_seed: ', self.random_seed,
            '   alpha: ', self.alpha,
            '   beta: ', self.beta,
            '   gamma: ', self.gamma,
            '   z_dim: ', self.z_dim,
            '   batch_size: ', self.batch_size,
        )

        # 导入图片数据并处理

        img_path = glob.glob(self.datasetPATH)

        dataset, img_shape = image_dataset(img_path, self.batch_size, 64)
        print(dataset, img_shape)
        db_iter = iter(dataset.repeat())

        G1 = Generator()
        G1.build(input_shape=(None, self.z_dim))

        G2 = Generator()
        G2.build(input_shape=(None, self.z_dim))

        G3 = Generator()
        G3.build(input_shape=(None, self.z_dim))

        D1 = Discriminator()
        D1.build(input_shape=(None, 64, 64, 3))

        D2 = Discriminator()
        D2.build(input_shape=(None, 64, 64, 3))

        D3 = Discriminator()
        D3.build(input_shape=(None, 64, 64, 3))

        d_loss_share_val = 0
        d1_loss_share_force = 0
        d2_loss_share_force = 0
        d3_loss_share_force = 0
        g_loss_share_val = 0
        g1_loss_share_force = 0
        g2_loss_share_force = 0
        g3_loss_share_force = 0

        Dloss_list = []
        D1loss_list = []
        D2loss_list = []
        D3loss_list = []

        Gloss_list = []
        G1loss_list = []
        G2loss_list = []
        G3loss_list = []

        epoch_list=[]


        # 优化器
        G1optimizer = tf.optimizers.Adam(learning_rate=self.G1_learning_rate, beta_1=0.5)
        G2optimizer = AdaBeliefOptimizer(learning_rate=self.G2_learning_rate, epsilon=1e-12, rectify=False,print_change_log=False)
        G3optimizer = tf.optimizers.Adam(learning_rate=self.G3_learning_rate, beta_1=0.5)

        D1optimizer = tf.optimizers.Adam(learning_rate=self.D1_learning_rate, beta_1=0.5)
        # pip install adabelief-tf==0.2.0
        # https://github.com/juntang-zhuang/Adabelief-Optimizer
        D2optimizer = AdaBeliefOptimizer(learning_rate=self.D2_learning_rate, epsilon=1e-12, rectify=False,print_change_log=False)
        D3optimizer = tf.optimizers.Adam(learning_rate=self.D3_learning_rate, beta_1=0.5)



        for epoch in range(self.epochs):
            batch_z = tf.random.uniform([self.batch_size, self.z_dim], minval=-1, maxval=1)
            batch_x = next(db_iter)
            g1Image = G1(batch_z, is_training)
            g2Image = G2(batch_z, is_training)
            g3Image = G3(batch_z, is_training)

            # 训练辨别器1
            with tf.GradientTape() as tape:  # 梯度带， 求导
                d1_loss_temp = d_loss_fn(g1Image, g2Image, g3Image, D1, batch_x, is_training, is_wgan=False)
                d1_loss = self.alpha * d1_loss_temp + d_loss_share_val + d1_loss_share_force
                grads_d1 = tape.gradient(d1_loss, D1.trainable_variables)
                D1optimizer.apply_gradients(zip(grads_d1, D1.trainable_variables))

            # 训练辨别器2
            with tf.GradientTape() as tape:  # 梯度带， 求导
                d2_loss_temp = d_loss_fn(g1Image, g2Image, g3Image, D2, batch_x, is_training, is_wgan=False)
                d2_loss = self.beta * d2_loss_temp + d_loss_share_val + d2_loss_share_force
                grads_d2 = tape.gradient(d2_loss, D2.trainable_variables)
                D2optimizer.apply_gradients(zip(grads_d2, D2.trainable_variables))

            # 训练辨别器3
            with tf.GradientTape() as tape:  # 梯度带， 求导
                d3_loss_temp = d_loss_fn(g1Image, g2Image, g3Image, D3, batch_x, is_training, is_wgan=False)
                d3_loss = self.gamma * d3_loss_temp + d_loss_share_val + d3_loss_share_force
                grads_d3 = tape.gradient(d3_loss, D3.trainable_variables)
                D3optimizer.apply_gradients(zip(grads_d3, D3.trainable_variables))

            # 训练生成器1
            with tf.GradientTape() as tape:
                g1_loss_temp = g_loss_fn(G1, D1, D2, D3, batch_z, is_training)
                g1_loss = self.alpha * g1_loss_temp + g_loss_share_val + g1_loss_share_force
                grads_g1 = tape.gradient(g1_loss, G1.trainable_variables)
                G1optimizer.apply_gradients(zip(grads_g1, G1.trainable_variables))

            # 训练生成器2
            with tf.GradientTape() as tape:
                g2_loss_temp = g_loss_fn(G2, D1, D2, D3, batch_z, is_training)
                g2_loss = self.alpha * g2_loss_temp + g_loss_share_val + g2_loss_share_force
                grads_g2 = tape.gradient(g2_loss, G2.trainable_variables)
                G2optimizer.apply_gradients(zip(grads_g2, G2.trainable_variables))

            # 训练生成器3
            with tf.GradientTape() as tape:
                g3_loss_temp = g_loss_fn(G3, D1, D2, D3, batch_z, is_training)
                g3_loss = self.alpha * g3_loss_temp + g_loss_share_val + g3_loss_share_force
                grads_g3 = tape.gradient(g3_loss, G3.trainable_variables)
                G3optimizer.apply_gradients(zip(grads_g3, G3.trainable_variables))

            d_loss = (d1_loss + d2_loss + d3_loss) / 3
            g_loss = (g1_loss + g2_loss + g3_loss) / 3

            # condition share loss D
            if d1_loss - (d2_loss + d3_loss) / 2 > 1:
                d_loss_share_val = 0.1 * (d1_loss - (d2_loss + d3_loss) / 2)
            elif d2_loss - (d3_loss + d1_loss) / 2 > 1:
                d_loss_share_val = 0.1 * (d2_loss - (d3_loss + d1_loss) / 2)
            elif d3_loss - (d2_loss + d1_loss) / 2 > 1:
                d_loss_share_val = 0.1 * (d3_loss - (d2_loss + d1_loss) / 2)
            else:
                d_loss_share_val = 0

            # force share loss D
            d1_loss_share_force = (1 - self.alpha) / 2 * d2_loss + (1 - self.alpha) / 2 * d3_loss
            d2_loss_share_force = (1 - self.beta) / 2 * d1_loss + (1 - self.beta) / 2 * d3_loss
            d3_loss_share_force = (1 - self.gamma) / 2 * d1_loss + (1 - self.gamma) / 2 * d2_loss

            # condition share loss G
            if g1_loss - (g2_loss + g3_loss) / 2 > 1:
                g_loss_share_val = 0.1 * (g1_loss - (g2_loss + g3_loss) / 2)
            elif g2_loss - (g3_loss + g1_loss) / 2 > 1:
                g_loss_share_val = 0.1 * (g2_loss - (g3_loss + g1_loss) / 2)
            elif g3_loss - (g2_loss + g1_loss) / 2 > 1:
                g_loss_share_val = 0.1 * (g3_loss - (g2_loss + g1_loss) / 2)
            else:
                g_loss_share_val = 0

            # force share loss G
            g1_loss_share_force = (1 - self.alpha) / 2 * g2_loss + (1 - self.alpha) / 2 * g3_loss
            g2_loss_share_force = (1 - self.beta) / 2 * g1_loss + (1 - self.beta) / 2 * g3_loss
            g3_loss_share_force = (1 - self.gamma) / 2 * g1_loss + (1 - self.gamma) / 2 * g2_loss

            if epoch % 10 == 0:
                print(
                    epoch, '\n',
                    'condition_d_loss_share_val %.3f' % d_loss_share_val,
                    'd1_loss_share_force %.3f' % d1_loss_share_force,
                    'd2_loss_share_force %.3f' % d2_loss_share_force,
                    'd3_loss_share_force %.3f \n' % d3_loss_share_force,
                    'condition_g_loss_share_val %.3f' % g_loss_share_val,
                    'g1_loss_share_force %.3f' % g1_loss_share_force,
                    'g2_loss_share_force %.3f' % g2_loss_share_force,
                    'g3_loss_share_force %.3f \n' % g3_loss_share_force
                )

            # 每100次 打印一次损失用来观察
            if epoch % 100 == 0:
                print(epoch, '\n',
                      'D1_loss %.5f' % d1_loss,
                      'D2_loss %.5f' % d2_loss,
                      'D3_loss %.5f \n' % d3_loss,
                      'G1_loss %.5f' % g1_loss,
                      'G2_loss %.5f' % g2_loss,
                      'G3_loss %.5f\n' % g3_loss,
                      'D_loss %.5f' % d_loss,
                      'G_loss %.5f\n' % g_loss

                      )

                Dloss_list.append (d_loss)
                D1loss_list.append (d1_loss)
                D2loss_list.append (d2_loss)
                D3loss_list.append (d3_loss)

                Gloss_list.append(g_loss)
                G1loss_list.append(g1_loss)
                G2loss_list.append(g2_loss)
                G3loss_list.append(g3_loss)

                epoch_list.append(epoch)

                # 并保存图片
                z = tf.random.normal([25, self.z_dim])
                fake_image_g1 = G1(z, training=False)
                fake_image_g2 = G2(z, training=False)
                fake_image_g3 = G3(z, training=False)

                output = self.outputPATH
                img_path_g1 = os.path.join(output, '%d g1.png' % epoch)
                img_path_g2 = os.path.join(output, '%d g2.png' % epoch)
                img_path_g3 = os.path.join(output, '%d g3.png' % epoch)

                save_result(fake_image_g1.numpy(), 5, img_path_g1, color_mode='P')
                save_result(fake_image_g2.numpy(), 5, img_path_g2, color_mode='P')
                save_result(fake_image_g3.numpy(), 5, img_path_g3, color_mode='P')

                #loss cuve
                plt.plot(epoch_list, Dloss_list, "r", linewidth=2.0, linestyle='-', ms=10, label="D loss")
                plt.plot(epoch_list, Gloss_list, "b", linewidth=2.0, linestyle='-', ms=10, label="G loss")
                plt.xticks(epoch_list, rotation=10)
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.title("loss curve")
                plt.legend(loc="upper right")
                # plt.plot(x,y)
                plt.savefig("a.jpg")
                plt.show()