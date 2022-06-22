class Config():
    learning_rate = 0.0015
    adam_betas = (0.9, 0.999)
    epoch_count = 200
    batch_size = 256
    experiment_name = 'intent_classification_12'
    save_every = 25
    
    dataset_path = './pl-PL.jsonl'
    dataset_random_seed = 7