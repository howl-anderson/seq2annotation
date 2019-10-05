import copy
import os

import tensorflow as tf

from seq2annotation import utils
from seq2annotation.utils import class_from_module_path
from tokenizer_tools.hooks import TensorObserveHook


tf.logging.set_verbosity(tf.logging.INFO)

observer_hook = TensorObserveHook(
    {"fake_golden": "fake_golden:0", "fake_prediction": "fake_prediction:0"},
    {
        # "word_str": "word_strings_Lookup:0",
        "predictions_id": "predictions:0",
        "predict_str": "predict_Lookup:0",
        "labels_id": "labels:0",
        "labels_str": "IteratorGetNext:2",
    },
    {
        "word_str": lambda x: x.decode(),
        "predict_str": lambda x: x.decode(),
        "labels_str": lambda x: x.decode(),
    },
)


def train_model(train_inpf, eval_inpf, config, model_fn, model_name):
    estimator_params = copy.deepcopy(config)

    indices = [idx for idx, tag in enumerate(config["tags_data"]) if tag.strip() != "O"]
    num_tags = len(indices) + 1
    estimator_params["_indices"] = indices
    estimator_params["_num_tags"] = num_tags

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=config["save_checkpoints_secs"])

    model_specific_name = "{model_name}-{batch_size}-{learning_rate}-{max_steps}-{max_steps_without_increase}".format(
        model_name=model_name,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        max_steps_without_increase=config["max_steps_without_increase"],
    )

    instance_model_dir = os.path.join(config["model_dir"], model_specific_name)

    if config["use_tpu"]:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu=config["tpu_name"],
            zone=config["tpu_zone"],
            project=config["gcp_project"],
        )

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=instance_model_dir,
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True
            ),
            tpu_config=tf.contrib.tpu.TPUConfig(),
        )

        tpu_estimator_params = copy.deepcopy(estimator_params)
        # remove reserved keys
        # tpu_estimator_params['train_batch_size'] = tpu_estimator_params['batch_size']
        del tpu_estimator_params["batch_size"]
        # del tpu_estimator_params['context']

        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            params=tpu_estimator_params,
            config=run_config,
            use_tpu=True,
            train_batch_size=estimator_params["batch_size"],
            eval_batch_size=estimator_params["batch_size"],
            predict_batch_size=estimator_params["batch_size"],
        )
    else:
        estimator = tf.estimator.Estimator(
            model_fn, instance_model_dir, cfg, estimator_params
        )

    # Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    utils.create_dir_if_needed(estimator.eval_dir())

    # hook_params = params['hook']['stop_if_no_increase']
    # hook = tf.contrib.estimator.stop_if_no_increase_hook(
    #     estimator, 'f1',
    #     max_steps_without_increase=hook_params['max_steps_without_increase'],
    #     min_steps=hook_params['min_steps'],
    #     run_every_secs=hook_params['run_every_secs']
    # )

    # build hooks from config
    train_hook = []
    for i in config.get("train_hook", []):
        class_ = class_from_module_path(i["class"])
        params = i["params"]
        if i.get("inject_whole_config", False):
            params["config"] = config
        train_hook.append(class_(**params))

    eval_hook = []
    for i in config.get("eval_hook", []):
        class_ = class_from_module_path(i["class"])
        params = i["params"]
        if i.get("inject_whole_config", False):
            params["config"] = config
        eval_hook.append(class_(**params))

    if eval_inpf:
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_inpf, hooks=train_hook, max_steps=config["max_steps"]
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_inpf, throttle_secs=config["throttle_secs"], hooks=eval_hook
        )
        evaluate_result, export_results = tf.estimator.train_and_evaluate(
            estimator, train_spec, eval_spec
        )
    else:
        estimator.train(
            input_fn=train_inpf, hooks=train_hook, max_steps=config["max_steps"]
        )
        evaluate_result, export_results = {}, None

        # # Write predictions to file
    # def write_predictions(name):
    #     output_file = preds_file(name)
    #     with tf.io.gfile.GFile(output_file, 'w') as f:
    #         test_inpf = functools.partial(input_fn, fwords(name))
    #         golds_gen = generator_fn(fwords(name))
    #         preds_gen = estimator.predict(test_inpf)
    #         for golds, preds in zip(golds_gen, preds_gen):
    #             ((words, _), tags) = golds
    #             preds_tags = [i.decode() for i in preds['tags']]
    #             for word, tag, tag_pred in zip(words, tags, preds_tags):
    #                 # f.write(b' '.join([word, tag, tag_pred]) + b'\n')
    #                 f.write(' '.join([word, tag, tag_pred]) + '\n')
    #             # f.write(b'\n')
    #             f.write('\n')
    #
    # for name in ['train', 'test']:
    #     write_predictions(name)

    # export saved_model
    feature_spec = {
        # 'words': tf.placeholder(tf.int32, [None, None]),
        "words": tf.placeholder(tf.string, [None, None]),
        "words_len": tf.placeholder(tf.int32, [None]),
    }

    if config.get("forced_saved_model_dir"):
        instance_saved_dir = config.get("forced_saved_model_dir")
    else:
        instance_saved_dir = os.path.join(
            config["saved_model_dir"], model_specific_name
        )

    utils.create_dir_if_needed(instance_saved_dir)

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        feature_spec
    )
    raw_final_saved_model = estimator.export_saved_model(
        instance_saved_dir,
        serving_input_receiver_fn,
        # assets_extra={
        #     'tags.txt': 'data/tags.txt',
        #     'vocab.txt': 'data/unicode_char_list.txt'
        # }
    )

    final_saved_model = raw_final_saved_model.decode("utf-8")

    return evaluate_result, export_results, final_saved_model
