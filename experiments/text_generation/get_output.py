def undetectable_exp_pipeline(output_path, model_str, reweight_type, dataset_name):
    from .common import set_spawn

    set_spawn()

    from torch.multiprocessing import Process, Queue, Event

    from .common import get_num_gpus

    num_gpus = get_num_gpus()

    print("num_gpus:", num_gpus, flush=True)

    tq = Queue(maxsize=num_gpus)
    tqe = Event()
    rq = Queue()
    rqe = Event()

    from .common import batched_wp_task_worker, transformer_worker
    from . import get_in_ds_undetectable_exp

    task_worker_ = Process(
        target=batched_wp_task_worker,
        args=(tq,),
        kwargs={
            "get_in_ds": get_in_ds_undetectable_exp,
            "reweight_type": reweight_type,
            "dataset_name": dataset_name,
            "model_str": model_str,
            "batch_size": 1,
        },
    )
    gpu_workers = [
        Process(
            target=transformer_worker,
            args=(tq, tqe, rq, i),
            kwargs={
                "model_str": model_str,
                "decoder_only": True,
                "generation_kwargs": {
                    "max_new_tokens": 512,
                    "temperature": 1.0,
                },
                "tokenization_kwargs": {
                    "task_template": "{input}",
                    "max_length": 3072,
                },
            },
        )
        for i in range(num_gpus)
    ]
    from .common import simple_store_worker

    store_worker = Process(target=simple_store_worker, args=(output_path, rq, rqe))

    task_worker_.start()
    # exit(0)
    for w in gpu_workers:
        w.start()
    store_worker.start()

    task_worker_.join()
    assert task_worker_.exitcode == 0
    tqe.set()
    for w in gpu_workers:
        w.join()
        assert w.exitcode == 0
    rqe.set()
    store_worker.join()
    assert store_worker.exitcode == 0
