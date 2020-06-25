# coding: utf-8
"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
start_time = time.time()

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import csv
import cv2
import tensorflow as tf

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

slim = tf.contrib.slim

tf.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string('checkpoint_path_inception_v3', ' ', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_v4', ' ', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_resnet_v2', ' ', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet', ' ', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', ' ', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', ' ', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', ' ', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('input_dir', ' ', 'Input directory with images.')

tf.flags.DEFINE_string('mask_dir', ' ', 'Mask directory with images.')

tf.flags.DEFINE_string('output_dir', ' ', 'Output directory with images.')

tf.flags.DEFINE_string('dev_path', ' ', 'Input directory with images.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 20, 'Number of iterations.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 4, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')

FLAGS = tf.flags.FLAGS

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)

def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
        filenames: list file names without path of each image
            Lenght of this list could be less than batch_size, in this case only
            first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    files_mask = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width])
    filenames_image = []

    idx = 0
    batch_size = batch_shape[0]

    # Images for inception classifier are normalized to be in [-1, 1] interval.

    with open(os.path.join(FLAGS.dev_path), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath_image = os.path.join(input_dir, row['ImageId'] + '.png')
            try:
                with open(filepath_image, 'rb') as f:
                    raw_image = imread(f, mode='RGB').astype(np.float)
                    image = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width]) / 255.0
            except:
                continue
            images[idx, :, :, :] = image * 2.0 - 1.0
            filenames_image.append(os.path.basename(filepath_image))

            filepath_mask = os.path.join(FLAGS.mask_dir, row['ImageId'] + '.png')
            newmask = cv2.imread(filepath_mask, 0)
            mask = np.zeros(image.shape[:2], np.uint8)
            mask[newmask > 10] = 1
            files_mask[idx, :, :] = mask

            idx += 1
            if idx == batch_size:
                yield filenames_image, images, files_mask
                filenames_image = []
                images = np.zeros(batch_shape)
                idx = 0
        if idx > 0:
            yield filenames_image, images, files_mask


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5)#, format='png')


def graph(x, mask, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    alpha = eps / 10
    momentum = FLAGS.momentum
    num_classes = 1001

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False)
        logits_v3_rotated, end_points_v3_R = inception_v3.inception_v3(
            rotate(x), num_classes=num_classes, is_training=False, reuse=True)
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(x), num_classes=num_classes, is_training=False)
        logits_v4_rotated, end_points_v4_R = inception_v4.inception_v4(
            rotate(x), num_classes=num_classes, is_training=False, reuse=True)
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(x), num_classes=num_classes, is_training=False,reuse=True)
        logits_res_v2_rotated, end_points_res_v2_R = inception_resnet_v2.inception_resnet_v2(
            rotate(x), num_classes=num_classes, is_training=False, reuse=True)
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
            input_diversity(x), num_classes=num_classes, is_training=False)
        logits_resnet_rotated, end_points_resnet_R = resnet_v2.resnet_v2_152(
            rotate(x), num_classes=num_classes, is_training=False, reuse=True)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')
        logits_ens3_adv_v3_rotated, end_points_ens3_adv_v3_R = inception_v3.inception_v3(
            rotate(x), num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3', reuse=True)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')
        logits_ens4_adv_v3_rotated, end_points_ens4_adv_v3_R = inception_v3.inception_v3(
            rotate(x), num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3', reuse=True)

    # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    #     logits_ens_advres_v2, end_points_ens_advres_v2 = inception_resnet_v2.inception_resnet_v2(
    #         input_diversity(x), num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
    #     logits_ens_advres_v2_rotated, end_points_ens_advres_v2_R = inception_resnet_v2.inception_resnet_v2(
    #         rotate(x), num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2', reuse=True)



    logits = (logits_v3 + logits_v4 + logits_res_v2 + logits_resnet
              + logits_v3_rotated + logits_v4_rotated + logits_res_v2_rotated + logits_resnet_rotated
              + logits_ens3_adv_v3 + logits_ens4_adv_v3
              + logits_ens3_adv_v3_rotated + logits_ens4_adv_v3_rotated ) / 12

    auxlogits = (end_points_v3['AuxLogits'] + end_points_v3_R['AuxLogits']
                + end_points_v4['AuxLogits'] + end_points_v4_R['AuxLogits']
                + end_points_res_v2['AuxLogits']+ end_points_res_v2_R['AuxLogits']
                + end_points_ens3_adv_v3['AuxLogits'] + end_points_ens3_adv_v3_R['AuxLogits']
                 + end_points_ens4_adv_v3['AuxLogits'] + end_points_ens4_adv_v3_R['AuxLogits']
                 # + end_points_ens_advres_v2['AuxLogits'] + end_points_ens_advres_v2_R['AuxLogits']
                 ) / 10

    cross_entropy = tf.losses.softmax_cross_entropy(y,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(y,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=0.4)
    noise = tf.gradients(cross_entropy, x)[0]

    noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')

    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise

    noise = noise * mask[:, :, :, np.newaxis]

    x = x + alpha * tf.sign(noise)

    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, mask, y, i, x_max, x_min, noise


def stop(x, mask, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def input_diversity(input_tensor):
    
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))

    # return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    proba = np.random.uniform(0, 1)
    if proba < FLAGS.prob:
        out = padded
    else:
        out = input_tensor
    return out
    # return padded

def rotate(input_tensor):
    rotated = tf.contrib.image.rotate(input_tensor, tf.random_uniform((), minval=-np.pi / 18, maxval=np.pi / 18))
    return rotated


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    print(time.time() - start_time)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_mask = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width])

        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)
        y = tf.one_hot(predicted_labels, num_classes)

        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, x_mask, y, i, x_max, x_min, grad])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

        with tf.Session() as sess:
            s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
            s3.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
            s4.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
            s5.restore(sess, FLAGS.checkpoint_path_inception_v4)
            s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
            # s7.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
            s8.restore(sess, FLAGS.checkpoint_path_resnet)

            print(time.time() - start_time)

            for filenames, images, files_mask in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images,  x_mask: files_mask})
                save_images(adv_images, filenames, FLAGS.output_dir)

        print(time.time() - start_time)


if __name__ == '__main__':
    tf.app.run()
