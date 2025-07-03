def pipeline(output_path, score_save_path, eps, model_str, dataset_name):
    # raise NotImplementedError
    from .common import set_spawn

    set_spawn()

    from torch.multiprocessing import Process, Queue, Event

    from .common import get_num_gpus

    num_gpus = get_num_gpus()

    tq = Queue(maxsize=num_gpus)
    tqe = Event()
    rq = Queue()
    rqe = Event()
    r2q = Queue()
    r2qe = Event()

    from .common import (
        merged_task_worker,
        # score_worker,
        watermark_score_worker,
        remove_text_worker,
        simple_store_worker,
    )

    from . import get_in_ds_undetectable_exp

    task_worker_ = Process(
        target=merged_task_worker,
        args=(get_in_ds_undetectable_exp, output_path, tq),
        kwargs={"batch_size": 1, "dataset_name": dataset_name},
    )

    score_worker_ = [
        Process(
            target=watermark_score_worker,
            args=(tq, tqe, rq, i),
            kwargs={
                "oracle_model_str": model_str,
                "decoder_only": True,
                "eps": eps,
                "tokenization_kwargs": {
                    "task_template": "{input}",
                    "max_length": 3072,
                },
            },
        )
        for i in range(num_gpus)
    ]
    rt_worker = Process(target=remove_text_worker, args=(rq, rqe, r2q))
    store_worker = Process(
        target=simple_store_worker,
        args=(score_save_path, r2q, r2qe),
    )

    task_worker_.start()
    for w in score_worker_:
        w.start()
    rt_worker.start()
    store_worker.start()

    task_worker_.join()
    assert task_worker_.exitcode == 0
    tqe.set()
    for w in score_worker_:
        w.join()
        assert w.exitcode == 0
    rqe.set()
    rt_worker.join()
    assert rt_worker.exitcode == 0
    r2qe.set()
    store_worker.join()
    assert store_worker.exitcode == 0
