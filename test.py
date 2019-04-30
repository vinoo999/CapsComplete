import os
import time
from tensorflow.python.client import timeline
from importlib import import_module
from capslayer.data.datasets.mnist import DataLoader
from capslayer.plotlib import plot_activation
import tensorflow as tf
import numpy as np
import capslayer as cl
from .data.data_download import start_download

def save_to(is_training, results_dir='./results/logs'):
    os.makedirs(os.path.join(results_dir, "activations"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "timelines"), exist_ok=True)

    if is_training:
        loss = os.path.join(results_dir, 'loss.csv')
        train_acc = os.path.join(results_dir, 'train_acc.csv')
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
        fd = {"train_acc": fd_train_acc,
              "loss": fd_loss,
              "val_acc": fd_val_acc}
    else:
        test_acc = os.path.join(results_dir, 'test_acc.csv')
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        fd = {"test_acc": fd_test_acc}

    return(fd)

def train(model, data_loader, batch_size=4, n_gpus=1, train_sum_every=200,
          num_steps=50000, val_sum_every=500, save_ckpt_every=1000, 
          iter_routing=3, regularization_scale=0.392,
          logdir='./results/checkpoints', results_dir='./results/logs'):
    # Setting up model
    training_iterator = data_loader(batch_size, mode="train")
    validation_iterator = data_loader(batch_size, mode="eval")
    inputs = data_loader.next_element["images"]
    labels = data_loader.next_element["labels"]
    
    model.create_network(inputs, labels)

    loss, train_ops, summary_ops = model.train(n_gpus)

    # Creating files, saver and summary writer to save training results
    fd = save_to(is_training=True)
    summary_writer = tf.summary.FileWriter(logdir)
    summary_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)

    # Setting up training session
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print("\nNote: all of results will be saved to directory: " + results_dir)
        for step in range(1, num_steps):
            start_time = time.time()
            if step % train_sum_every == 0:
                _, loss_val, train_acc, summary_str = sess.run([train_ops,
                                                               loss,
                                                               model.accuracy,
                                                               summary_ops],
                                                               feed_dict={data_loader.handle: training_handle})
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                out_path = os.path.join(results_dir, "timelines/timeline_%d.json" % step)
                with open(out_path, "w") as f:
                    f.write(ctf)
                summary_writer.add_summary(summary_str, step)
                fd["loss"].write("{:d},{:.4f}\n".format(step, loss_val))
                fd["loss"].flush()
                fd["train_acc"].write("{:d},{:.4f}\n".format(step, train_acc))
                fd["train_acc"].flush()
            else:
                _, loss_val = sess.run([train_ops, loss], feed_dict={data_loader.handle: training_handle})
                # assert not np.isnan(loss_val), 'Something wrong! loss is nan...'

            if step % val_sum_every == 0:
                print("evaluating, it will take a while...")
                sess.run(validation_iterator.initializer)
                probs = []
                targets = []
                total_acc = 0
                n = 0
                while True:
                    try:
                        val_acc, prob, label = sess.run([model.accuracy, model.probs, labels], feed_dict={data_loader.handle: validation_handle})
                        probs.append(prob)
                        targets.append(label)
                        total_acc += val_acc
                        n += 1
                    except tf.errors.OutOfRangeError:
                        break
                probs = np.concatenate(probs, axis=0)
                targets = np.concatenate(targets, axis=0).reshape((-1, 1))
                avg_acc = total_acc / n
                path = os.path.join(os.path.join(results_dir, "activations"))
                plot_activation(np.hstack((probs, targets)), step=step, save_to=path)
                fd["val_acc"].write("{:d},{:.4f}\n".format(step, avg_acc))
                fd["val_acc"].flush()
            if step % save_ckpt_every == 0:
                saver.save(sess,
                           save_path=os.path.join(logdir, 'model.ckpt'),
                           global_step=step)

                duration = time.time() - start_time
                log_str = ' step: {:d}, loss: {:.3f}, train acc {:.3f}, val accuracy: {:.3f}, time: {:.3f} sec/step' \
                        .format(step, loss_val, train_acc, avg_acc, duration)
                print(log_str)

def evaluate(model, data_loader, batch_size=4, n_gpus=1, train_sum_every=200,
          num_steps=50000, val_sum_every=500, save_ckpt_every=1000, 
          iter_routing=3, regularization_scale=0.392,
          log_dir='./results/checkpoints', results_dir='./results/logs'):
    # Setting up model
    test_iterator = data_loader(batch_size, mode="test")
    inputs = data_loader.next_element["images"]
    labels = data_loader.next_element["labels"]
    model.create_network(inputs, labels)

    # Create files to save evaluating results
    fd = save_to(is_training=False)
    saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        test_handle = sess.run(test_iterator.string_handle())
        saver.restore(sess, tf.train.latest_checkpoint(log_dir))
        tf.logging.info('Model restored!')

        probs = []
        targets = []
        total_acc = 0
        n = 0
        while True:
            try:
                test_acc, prob, label = sess.run([model.accuracy, model.probs, labels], feed_dict={data_loader.handle: test_handle})
                probs.append(prob)
                targets.append(label)
                total_acc += test_acc
                n += 1
            except tf.errors.OutOfRangeError:
                break
        probs = np.concatenate(probs, axis=0)
        targets = np.concatenate(targets, axis=0).reshape((-1, 1))
        avg_acc = total_acc / n
        out_path = os.path.join(results_dir, 'prob_test.txt')
        np.savetxt(out_path, np.hstack((probs, targets)), fmt='%1.2f')
        print('Classification probability for each category has been saved to ' + out_path)
        fd["test_acc"].write(str(avg_acc))
        fd["test_acc"].close()
        out_path = os.path.join(results_dir, 'test_accuracy.txt')
        print('Test accuracy has been saved to ' + out_path)

if __name__ == '__main__': 
    height = 28
    width = 28
    channels = 1
    num_label = 10
    net = Model(height=height, width=width, channels=channels, num_label=num_label)
    data_loader = DataLoader(path='data/mnist/', splitting='TVT', num_works=8)
    train(net, data_loader)
