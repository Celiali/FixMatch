from Project.fixmatch import train
import os
import ignite.distributed as idist
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.utils import convert_tensor
from ignite.metrics import Accuracy, Precision, Recall


def create_trainer(train_step, data, model, cfg):
    unlabeled_loader_iter = iter(data.loader['unlabeled'])

    trainer = Engine(train_step)
    @trainer.on(Events.ITERATION_STARTED)
    def prepare_batch(e):
        labeled_batch = e.state.batch
        e.state.batch = {
            'labeled_batch': labeled_batch,
            'unlabeled_batch': next(unlabeled_loader_iter)
        }

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_cta_rates(e):
        # TODO: keep track of the police applied to each img to update rates
        
        print('%d-th iteration complete'%e.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_ema_model():
        # TODO:
        pass

    # ================== Evaluate on testset =================
    metrics = {
        "accuracy": Accuracy(),
    }

    if not (idist.has_xla_support and idist.backend() == idist.xla.XLA_TPU):
        metrics.update({
            "precision": Precision(average=False),
            "recall": Recall(average=False),
        })

    def sup_prepare_batch(batch, device, non_blocking):
        x = convert_tensor(batch["image"], device, non_blocking)
        y = convert_tensor(batch["target"], device, non_blocking)
        return x, y

    eval_kwargs = dict(
        metrics=metrics,
        prepare_batch=sup_prepare_batch,
        device=idist.device(),
        non_blocking=True,
    )

    evaluator = create_supervised_evaluator(model, **eval_kwargs)

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.EXPERIMENT.validate_every)
                | Events.STARTED
                | Events.COMPLETED
                )
    def evaluate():
        evaluator.run(data.loader['test'])

    # TODO: debug

    # ================= Release resources ===================
    @trainer.on(Events.COMPLETED)
    def release_all_resources():
        nonlocal unlabeled_loader_iter

        if unlabeled_loader_iter is not None:
            unlabeled_loader_iter = None

    return trainer
