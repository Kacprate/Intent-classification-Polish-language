class Config():
    learning_rate = 0.003
    adam_betas = (0.9, 0.999)
    epoch_count = 10
    batch_size = 64
    experiment_name = 'test'
    
    dataset_path = './pl-PL.jsonl'
    dataset_random_seed = 13