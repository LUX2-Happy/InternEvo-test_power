JOB_NAME = "13b_train"
DO_ALERT = False

SEQ_LEN = 2048
HIDDEN_SIZE = 5120
NUM_ATTENTION_HEAD = 40
MLP_RATIO = 8 / 3
NUM_LAYER = 40
VOCAB_SIZE = 32000

MODEL_ONLY_FOLDER = "local:llm_ckpts/xxxx"
SAVE_CKPT_FOLDER = "local:llm_ckpts"
LOAD_CKPT_FOLDER = "local:llm_ckpts/49"

CHECKPOINT_EVERY = 50
ckpt = dict(
    enable_save_ckpt=False,
    save_ckpt_folder=SAVE_CKPT_FOLDER,
    load_ckpt_folder="local:llm_ckpts/",
    load_ckpt_info=dict(path=MODEL_ONLY_FOLDER, content=("model",), ckpt_type="internlm"),
    auto_resume=True,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=True,
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",
    oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),
)

TRAIN_FOLDER = None
VALID_FOLDER = None
data = dict(
    seq_len=SEQ_LEN,
    micro_num=4,
    micro_bsz=1,
    valid_micro_num=4,
    valid_every=50,
    pack_sample_into_one=False,
    total_steps=50000,
    skip_batches="",
    rampup_batch_size="",
    min_length=50,
    train_folder=TRAIN_FOLDER,
    valid_folder=VALID_FOLDER,
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
)

grad_scaler = dict(
    fp16=dict(
        initial_scale=2**16,
        min_scale=1,
        growth_interval=1000,
    ),
    growth_factor=2,
    backoff_factor=0.5,
    max_scale=2**24,
    hysteresis=2,
)

hybrid_zero_optimizer = dict(
    overlap_sync_grad=True,
    overlap_sync_param=True,
    reduce_bucket_size=512 * 1024 * 1024,
    clip_grad_norm=1.0,
)

loss = dict(
    label_smoothing=0,
)

adam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,
    warmup_ratio=0.01,
    eta_min=1e-5,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

model = dict(
    checkpoint=False,
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    use_flash_attn=True,
    num_chunks=1,
)

parallel = dict(
    zero1=dict(size=-1, fsdp=False),
    tensor=4,
    pipeline=dict(size=1, interleaved_overlap=True),
    sequence_parallel=False,
)

cudnn_deterministic = False
cudnn_benchmark = False

monitor = dict(
    alert=dict(
        enable_feishu_alert=DO_ALERT,
        feishu_alert_address=None,
        light_monitor_address=None,
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(
        queue_max_length=10,
    ),
)
