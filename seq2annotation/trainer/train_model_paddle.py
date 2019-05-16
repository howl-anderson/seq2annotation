import os
import time

import paddle
import paddle.fluid as fluid


def train_model(train_inpf, eval_inpf, config, *args, **kwargs):
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program=main_program, startup_program=startup_program):
        words = fluid.layers.data(name='words', shape=[1], dtype='int64',
                                  lod_level=1)
        tags = fluid.layers.data(name='tags', shape=[1], dtype='int64',
                                 lod_level=1)

        embed_char = fluid.layers.embedding(
            input=words,
            size=[config['embedding_vocabulary_size'], config['embedding_dim']],
            dtype='float32')

        place = fluid.CPUPlace()

        train_reader = paddle.batch(
            paddle.reader.shuffle(train_inpf, buf_size=500),
            batch_size=config['batch_size'])

        forward_hidden_state, _ = fluid.layers.dynamic_lstm(
            input=embed_char,
            size=config['embedding_dim'],  # TODO: paddle don't support hidden_size != embedding size
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid')

        backward_hidden_state, _ = fluid.layers.dynamic_lstm(
            input=embed_char,
            size=config['embedding_dim'],
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=True)

        hidden_state = fluid.layers.concat(
            [forward_hidden_state, backward_hidden_state], 1)

        feature = fluid.layers.dropout(hidden_state, 0.1)

        score = fluid.layers.fc(feature, len(config['tags_data']))

        crf_cost = fluid.layers.linear_chain_crf(
            input=score,
            label=tags,
            param_attr=fluid.ParamAttr(name='crfw')
        )

        avg_cost = fluid.layers.mean(crf_cost)

        crf_decode = fluid.layers.crf_decoding(
            input=score,
            param_attr=fluid.ParamAttr(name='crfw')
        )

        sgd_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.01)

        sgd_optimizer.minimize(avg_cost)

        feeder = fluid.DataFeeder(place=place, feed_list=[words, tags])

        exe = fluid.Executor(place)

        exe.run(startup_program)

        save_dirname = os.path.join(config['saved_model_dir'], str(int(time.time())))

        for pass_id in range(config['epochs']):
            print(">>> pass_id: {}".format(pass_id))
            for data in train_reader():
                # print(data)
                feed = feeder.feed(data)

                avg_loss_value, = exe.run(
                    main_program, feed=feed, fetch_list=[avg_cost],
                    return_numpy=True)
                print(avg_loss_value[0])

        if save_dirname is not None:
            fluid.io.save_inference_model(save_dirname, ['words'], [crf_decode], exe)

            # save asset
            def write_asset(output_file, data):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'wt') as fd:
                    fd.write('\n'.join(data))

            write_asset(os.path.join(save_dirname, 'data/vocabulary.txt'), config['vocab_data'])
            write_asset(os.path.join(save_dirname, 'data/tags.txt'), config['tags_data'])

        return None, None, save_dirname
