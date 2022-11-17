from torch.utils.data import DataLoader, RandomSampler, BatchSampler


def get_dataloaders(
    train_set,
    test_set,
    args,
    seed_worker=None,
    g=None,
):
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, test_loader


def get_samplers(
    train_set,
    test_set,
    args,
    seed_worker=None,
    g=None,
):
    train_sampler = RandomSampler(data_source=train_set,
                                  replacement=True,
                                  num_samples=args.updates * args.batch_size,
                                  generator=g)
    test_sampler = RandomSampler(data_source=test_set,
                                 replacement=True,
                                 num_samples=args.updates * args.batch_size // 5,
                                 generator=g)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.num_workers,
                              worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             sampler=test_sampler,
                             num_workers=args.num_workers,
                             worker_init_fn=seed_worker)

    return train_loader, test_loader