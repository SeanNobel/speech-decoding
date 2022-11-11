import torch


def get_dataloaders(
    train_set,
    test_set,
    args,
    seed_worker=None,
    g=None,
):

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, test_loader