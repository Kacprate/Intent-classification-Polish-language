class Config():
    learning_rate = 0.003
    adam_betas = (0.9, 0.999)
    epoch_count = 100
    batch_size = 64
    experiment_name = 'intent_classification_7'
    save_every = 25
    
    dataset_path = './pl-PL.jsonl'
    dataset_random_seed = 13