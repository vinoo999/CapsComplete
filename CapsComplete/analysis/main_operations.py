from CapsComplete.analysis.visualization import plot_imgs
from CapsComplete.data.preprocess import load_data, get_batch, occlude
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

def train(model, num_label, 
          dataset='mnist', 
          batch_size=128, n_epochs=20,
          results_dir='./results/', log_dir='./logs/',
          train_sum_freq=100,
          val_sum_freq=500,
          save_freq=3,
          num_samples=-1,
          occlusion=0,
          ox_dim=0, oy_dim=0, occ_prob=0, dataset_path=None):
    
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(dataset, batch_size, is_training=True, quantity=num_samples, occlusion=occlusion, ox_dim=ox_dim, oy_dim=oy_dim, occ_prob=occ_prob, path=dataset_path)

    fd_train_acc, fd_loss, fd_val_acc = save_to(results_dir, True)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=model.graph) as sess:
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        sess.run(tf.variables_initializer(model.graph.get_collection('variables')))
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("\nNote: all of results will be saved to directory: " + results_dir)

        batch_num = 0
        for epoch in range(n_epochs):
            print("Training for epoch %d/%d:" % (epoch, n_epochs))
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * batch_size
                end = start + batch_size
                global_step = epoch * num_tr_batch + step
                X_batch, Y_batch, batch_num = get_batch(trX, trY, batch_size, batch_num)
                if global_step % train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary], feed_dict={model.inputs: X_batch, model.labels: Y_batch})
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op, feed_dict={model.inputs: X_batch, model.labels: Y_batch})

                if val_sum_freq != 0 and (global_step) % val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * batch_size
                        end = start + batch_size
                        acc = sess.run(model.accuracy, {model.inputs: valX[start:end], model.labels: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()
            if (epoch + 1) % save_freq == 0:
                tf.logging.info('Model Saving')
                saver.save(sess, log_dir + 'epoch_' + str(epoch))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()

def evaluation(model, num_label, dataset='mnist', 
               batch_size=128,
               results_dir='./results/', log_dir='./logs/', 
               occlusion=0, ox_dim=0, oy_dim=0, occ_prob=0, dataset_path=None, model_restore=None):
    teX, teY, num_te_batch = load_data(dataset, batch_size, is_training=False,
                                        occlusion=occlusion, ox_dim=ox_dim, oy_dim=oy_dim, occ_prob=occ_prob, path=dataset_path)
    fd_test_acc = save_to(results_dir, False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=model.graph) as sess:
        saver = tf.train.Saver()
        if model_restore is None:
            saver.restore(sess, tf.train.latest_checkpoint(log_dir))
        else:
            saver.restore(sess, model_restore)
        tf.logging.info('Model restored!')

        test_acc = 0
        recon_error = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * batch_size
            end = start + batch_size
            acc, recons = sess.run([model.accuracy, model.recons], {model.inputs: teX[start:end], model.labels: teY[start:end]})
            test_acc += acc
            recon_error += np.sum(np.sqrt(np.sum((teX[start:end] - recons) ** 2, axis=(1,2,3) )))

        test_acc = test_acc / (batch_size * num_te_batch)
        recon_erorr = recon_error / (batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc) + ',' + str(recon_error))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + results_dir + '/test_acc.csv')
    return test_acc, recon_error

def self_reconstruction(model, num_labels, img_shape, iter=5, model_restore=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    maximized_imgs = None
    recon_imgs_on_max = None
    with tf.Session(config=config, graph=model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_restore)
        grad = tf.gradients(model.classification_loss, [model.inputs])
        train_step = model.inputs - tf.multiply(grad[0],1)
        for label in range(num_labels):
            # print("Label: ", label)
            input_imgs = np.zeros(img_shape)
            for i in range(iter): 
                input_imgs, probs = sess.run([train_step, model.probs], 
                                             feed_dict={model.inputs: input_imgs, model.labels: [label]})
                # if i % 10 == 0:
                #     print(probs.flatten()[label])
            # plot_imgs(input_imgs)
            probs, recons = sess.run([model.probs, model.recons], 
                                             feed_dict={model.inputs: input_imgs, model.labels: [label]})
            # print(probs.flatten()[label])
            # plot_imgs(recons)
            if maximized_imgs is None:
                maximized_imgs = input_imgs
            else:
                maximized_imgs = np.concatenate((maximized_imgs, input_imgs))
            if recon_imgs_on_max is None:
                recon_imgs_on_max = recons
            else:
                recon_imgs_on_max = np.concatenate((recon_imgs_on_max, recons))
        
        return maximized_imgs, recon_imgs_on_max

def get_reconstruction(model, imgs, labels=None, model_restore=None, occlusion=0, ox_dim=0, oy_dim=0, occ_prob=0):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_restore)
        occluded_imgs = occlude(imgs, occlusion=occlusion, ox_dim=ox_dim, oy_dim=oy_dim, occ_prob=occ_prob)
        recons, preds = sess.run([model.recons, model.predictions], 
                        feed_dict={model.inputs: occluded_imgs})
        
        return occluded_imgs, recons, preds


def get_reconstruction_w_gradient(model, imgs, n_steps=10, labels=None, model_restore=None, occlusion=0, ox_dim=0, oy_dim=0, occ_prob=0, lr=0.1):
    '''Fix this implementation to be better later'''
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_restore)
        
        grad = tf.gradients(model.classification_loss, [model.inputs])
        train_step = model.inputs - lr*tf.multiply(grad[0],1)
        predictions = sess.run(model.predictions, 
                          feed_dict={model.inputs: imgs})
        
        out_imgs = None
        occ_imgs = occlude(imgs, occlusion=occlusion, ox_dim=ox_dim, oy_dim=oy_dim, occ_prob=occ_prob)
        for i in range(imgs.shape[0]):
            new_img = occ_imgs[i,...][np.newaxis]
            prediction = predictions[i]
            for _ in range(n_steps):
                new_img = sess.run(train_step, 
                                             feed_dict={model.inputs: new_img, model.labels: [prediction]})
            out_img = sess.run(model.recons, 
                          feed_dict={model.inputs: new_img})
            if out_imgs is None:
                out_imgs = out_img
            else:
                out_imgs = np.concatenate((out_imgs, out_img))
        
        recon_error = np.sum(np.sqrt(np.sum((imgs - out_imgs) ** 2, axis=(1,2,3) )))
        return occ_imgs, out_imgs, recon_error



def save_to(results_dir, is_training):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if is_training:
        loss = os.path.join(results_dir,'loss.csv')
        train_acc = os.path.join(results_dir , 'train_acc.csv')
        val_acc = os.path.join(results_dir, 'val_acc.csv')

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return fd_train_acc, fd_loss, fd_val_acc
    else:
        test_acc = os.path.join(results_dir, 'test_acc.csv')
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return fd_test_acc