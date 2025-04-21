from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir="./dataset",
    epochs=150,
    batch_size=8,
    grad_accum_steps=2,
    lr=2e-4,
    lr_scheduler="cosine",
    warmup_epochs=10,
    use_amp=True,
    output_dir="./logs",
    save_best=True,
    early_stopping_patience=15,
    validate_every=1,
    device="cuda"
)