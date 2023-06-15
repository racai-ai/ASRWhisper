
class Config:
    #learning_rate = 0.0005
    learning_rate = 0.000005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 2
    num_worker = 2
    num_train_epochs = 11
    gradient_accumulation_steps = 4
    sample_rate = 16000
