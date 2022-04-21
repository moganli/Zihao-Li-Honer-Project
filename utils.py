import tensorflow as tf


def celoss_one(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zero(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def d_loss_fn(g1Image, g2Image, g3Image, D, batch_x, training, is_wgan):
    d1_g1_fake_logits = D(g1Image, training)
    d1_g2_fake_logits = D(g2Image, training)
    d1_g3_fake_logits = D(g3Image, training)

    d1_real_logits = D(batch_x, training)

    d1_g1_loss_fake = celoss_zero(d1_g1_fake_logits)
    d1_g2_loss_fake = celoss_zero(d1_g2_fake_logits)
    d1_g3_loss_fake = celoss_zero(d1_g3_fake_logits)
    
    #https://baike.baidu.com/item/%E5%9B%9B%E5%88%86%E4%BD%8D%E8%B7%9D/10671363?fr=aladdin
    #四分卫距离 IQR
    fakelist = [d1_g1_loss_fake, d1_g2_loss_fake, d1_g3_loss_fake]
    fakelist.sort()
    fake_discrepancy = fakelist[2] - fakelist[0]
    if fake_discrepancy > 1:
        d1_loss_fake = 0.8 * fakelist[1] + 0.1 * fakelist[2] + 0.1 * fakelist[0]
    else:
        d1_loss_fake = (d1_g1_loss_fake + d1_g2_loss_fake + d1_g3_loss_fake) / 3

    d1_loss_real = celoss_one(d1_real_logits)

    if is_wgan:
        gradientPenalty = gradient_Penalty(D, batch_x, g1Image)
        loss = d1_loss_real + d1_loss_fake + 1. * gradientPenalty
    else:
        loss = d1_loss_real + d1_loss_fake
    return loss


def g_loss_fn(G, D1, D2, D3, batch_z, training):
    fake_image = G(batch_z, training)
    d1_fake_logits = D1(fake_image, training)
    d2_fake_logits = D2(fake_image, training)
    d3_fake_logits = D3(fake_image, training)

    loss1 = celoss_one(d1_fake_logits)
    loss2 = celoss_one(d2_fake_logits)
    loss3 = celoss_one(d3_fake_logits)

    losslist = [loss1, loss2, loss3]
    losslist.sort()
    fake_discrepancy = losslist[2] - losslist[0]
    if fake_discrepancy > 1:
        loss = 0.8 * losslist[1] + 0.1 * losslist[2] + 0.1 * losslist[0]
    else:
        loss = (loss1 + loss2 + loss3) / 3

    return loss


def gradient_Penalty(D1, batch_x, gImage):
    batchsize = batch_x.shape[0]
    r = tf.random.uniform([batchsize, 1, 1, 1])

    r = tf.broadcast_to(r, batch_x.shape)

    interplate = r * batch_x + (1 - r) * gImage

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        Dlogits = D1(interplate)
    grads = tape.gradient(Dlogits, interplate)
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)
    gp = tf.reduce_mean((gp - 1) ** 2)

    return gp

