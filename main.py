import argparse
from SwarmGAN import SwarmGAN
import platform
import sys
import tensorflow as tf

print(sys.argv[0])
if platform.system().lower() == 'windows':
    print("windows")
elif platform.system().lower() == 'linux':
    print("linux")

print('tensorflow', tf.__version__)

FLAGS = None


def main():
    model = SwarmGAN(
        datasetPATH=FLAGS.datasetPATH,
        outputPATH=FLAGS.outputPATH,
        z_dim=FLAGS.z_dim,
        batch_size=FLAGS.batch_size,
        D1_learning_rate=FLAGS.D1_learning_rate,
        D2_learning_rate=FLAGS.D2_learning_rate,
        D3_learning_rate=FLAGS.D3_learning_rate,
        G1_learning_rate=FLAGS.G1_learning_rate,
        G2_learning_rate=FLAGS.G2_learning_rate,
        G3_learning_rate=FLAGS.G3_learning_rate,
        epochs=FLAGS.epochs,
        disp_freq=FLAGS.disp_freq,
        alpha=FLAGS.alpha,
        beta=FLAGS.beta,
        gamma=FLAGS.gamma,
        random_seed=1216)
    model.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetPATH', type=str, default='../dataset/anime-images/*.jpg',
                        help='input path: formal like /path1/path2/*.png')
        #../dataset/anime-images/*.jpg
        #../dataset/cifar10/cifar10/train/airplane/*.png
        #../dataset/mix/multi_domain_human_face_classification_dataset/celeb/*.jpg
        #../dataset/STL10/*.png
        #../dataset/tiny_imageNet/TinyImageNet/train/1/*.jpg
        #../dataset/mix/multi_domain_human_face_classification_dataset/cartoon/*.png
    parser.add_argument('--outputPATH', type=str, default='../generateImages/anime/gimages_anime_swarm_v3',
                        help='output path: formal like /path1/path2')
        #../generateImages/anime/gimages_anime_swarm_v2
        #../generateImages/cifar10_airplane/gimages_cifar10_airplane_swarm_v2
        #../generateImages/humanface/gimages_humanface_swarm_v2
        #../generateImages/STL10/gimages_STL10_swarm_v2
        #../generateImages/tiny_imageNet_1/gimages_tiny_imageNet_1_swarm_v2
        #../generateImages/cartoon/gimages_cartoon_swarm_v2
    parser.add_argument('--z_dim', type=int, default=100,
                        help='z dim')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size.')
    parser.add_argument('--D1_learning_rate', type=float, default=0.002,
                        help='first Discriminator Learning rate.')
    parser.add_argument('--D2_learning_rate', type=float, default=0.0002,
                        help='second Discriminator Learning rate.')
    parser.add_argument('--D3_learning_rate', type=float, default=0.006,
                        help='third Discriminator Learning rate.')                        
    parser.add_argument('--G1_learning_rate', type=float, default=0.002,
                        help='first generator Learning rate.')
    parser.add_argument('--G2_learning_rate', type=float, default=0.0002,
                        help='second generator Learning rate.')
    parser.add_argument('--G3_learning_rate', type=float, default=0.006,
                        help='third generator Learning rate.')
    parser.add_argument('--epochs', type=int, default=25000,
                        help='Number of epochs.')
    parser.add_argument('--disp_freq', type=int, default=100,
                        help='display frequency/save image frequency.')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='D1&G1 loss share (1-alpha)')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='D2&G2 loss share (1-beta)')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='D3&G3 loss share (1-gamma)')
    FLAGS, unparsed = parser.parse_known_args()
    main()
