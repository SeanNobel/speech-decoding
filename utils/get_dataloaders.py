from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from data.brennan2018 import CustomBatchSampler


def get_dataloaders(train_set, test_set, args, g, seed_worker):

    if args.use_sampler:
        train_batch_sampler = CustomBatchSampler(train_set, args)
        test_batch_sampler = CustomBatchSampler(test_set, args)
    else:
        train_batch_sampler = None
        test_batch_sampler = None

    if args.use_sampler:
        train_loader = DataLoader(
            train_set,
            shuffle=False,
            batch_sampler=train_batch_sampler,
            num_workers=6,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        test_loader = DataLoader(
            test_set,
            shuffle=False,
            batch_sampler=test_batch_sampler,
            num_workers=6,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        train_loader = DataLoader(
            train_set,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=6,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
        test_loader = DataLoader(
            test_set,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=6,
            pin_memory=False,
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